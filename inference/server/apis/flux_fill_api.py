#!/usr/bin/env python3
"""
FLUX.1-Fill-dev 图像填充API服务器
基于FluxFillPipeline的GPU管理器，提供图像填充REST API接口
"""

import os
import sys
import time
import base64
import io
import json
import random
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.multiprocessing as mp
from multiprocessing import Process, Queue, Event
import atexit
import signal
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# FastAPI相关导入
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from PIL import Image

# 设置多进程启动方法
mp.set_start_method('spawn', force=True)

# 添加src/examples到路径，以便导入tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'examples'))
try:
    from tools.prompt_utils import rewrite
except ImportError:
    print("警告: 无法导入prompt_utils，提示词增强功能将不可用")
    def rewrite(prompt):
        return prompt

from diffusers import FluxFillPipeline
import torch

# 配置参数
# Please replace with your own model path
model_repo_id = "/path/to/FLUX.1-Fill-dev"  # Replace with your FLUX.1-Fill-dev model path

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

NUM_GPUS_TO_USE = int(os.environ.get("NUM_GPUS_TO_USE", torch.cuda.device_count()))  
TASK_QUEUE_SIZE = int(os.environ.get("TASK_QUEUE_SIZE", 100))  
TASK_TIMEOUT = int(os.environ.get("TASK_TIMEOUT", 600))  # Fill任务可能需要更长时间

print(f"配置信息: 使用 {NUM_GPUS_TO_USE} 个GPU，队列大小 {TASK_QUEUE_SIZE}，超时时间 {TASK_TIMEOUT} 秒")


# ============== GPU管理器类（适配FLUX Fill图像填充） ==============
class FluxFillGPUWorker:
    def __init__(self, gpu_id, model_repo_id, task_queue, result_queue, stop_event):
        self.gpu_id = gpu_id
        self.model_repo_id = model_repo_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.device = f"cuda:{gpu_id}"
        self.pipe = None
        
    def initialize_model(self):
        """在指定GPU上初始化FLUX Fill图像填充模型"""
        try:
            torch.cuda.set_device(self.gpu_id)
            
            if not os.path.exists(self.model_repo_id):
                print(f"错误: 模型路径不存在 {self.model_repo_id}")
                return False
            
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            self.pipe = FluxFillPipeline.from_pretrained(
                self.model_repo_id, 
                torch_dtype=torch_dtype,
                local_files_only=True
            )
            self.pipe = self.pipe.to(self.device)
            print(f"GPU {self.gpu_id} FLUX Fill图像填充模型初始化成功")
            return True
        except Exception as e:
            print(f"GPU {self.gpu_id} FLUX Fill图像填充模型初始化失败: {e}")
            return False
    
    def process_task(self, task):
        """处理单个填充任务"""
        try:
            task_id = task['task_id']
            image_bytes = task['image_bytes']
            mask_bytes = task['mask_bytes']
            image = bytes_to_image(image_bytes)
            mask_image = bytes_to_image(mask_bytes)
            prompt = task['prompt']
            seed = task['seed']
            width = task['width']
            height = task['height']
            guidance_scale = task['guidance_scale']
            num_inference_steps = task['num_inference_steps']
            max_sequence_length = task['max_sequence_length']
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            with torch.cuda.device(self.gpu_id):
                with torch.inference_mode():
                    output = self.pipe(
                        prompt=prompt,
                        image=image,
                        mask_image=mask_image,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        max_sequence_length=max_sequence_length,
                        generator=generator,
                    ).images[0]
            
            return {
                'task_id': task_id,
                'image': output,
                'success': True,
                'gpu_id': self.gpu_id
            }
        except Exception as e:
            return {
                'task_id': task_id,
                'success': False,
                'error': str(e),
                'gpu_id': self.gpu_id
            }
    
    def run(self):
        """Worker主循环"""
        if not self.initialize_model():
            return
        
        print(f"GPU {self.gpu_id} FLUX Fill worker 启动")
        
        while not self.stop_event.is_set():
            try:
                # 从任务队列获取任务，设置超时以检查停止事件
                task = self.task_queue.get(timeout=1)
                if task is None:  # 毒丸，退出信号
                    break
                
                # 处理任务
                result = self.process_task(task)
                
                # 将结果放入结果队列
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"GPU {self.gpu_id} FLUX Fill worker 异常: {e}")
                continue
        
        print(f"GPU {self.gpu_id} FLUX Fill worker 停止")


# 全局GPU worker函数，用于spawn模式
def flux_fill_gpu_worker_process(gpu_id, model_repo_id, task_queue, result_queue, stop_event):
    worker = FluxFillGPUWorker(gpu_id, model_repo_id, task_queue, result_queue, stop_event)
    worker.run()


class MultiGPUFluxFillManager:
    def __init__(self, model_repo_id, num_gpus=None, task_queue_size=100):
        self.model_repo_id = model_repo_id
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.task_queue = Queue(maxsize=task_queue_size)  
        self.result_queue = Queue()  
        self.stop_event = Event()
        self.workers = []
        self.worker_processes = []
        self.task_counter = 0
        self.pending_tasks = {}  
        self.pending_tasks_lock = threading.Lock()
        
        print(f"初始化多GPU FLUX Fill图像填充管理器，使用 {self.num_gpus} 个GPU，队列大小 {task_queue_size}")
        
    def start_workers(self):
        """启动所有GPU workers"""
        for gpu_id in range(self.num_gpus):
            process = Process(target=flux_fill_gpu_worker_process, 
                            args=(gpu_id, self.model_repo_id, self.task_queue, 
                                  self.result_queue, self.stop_event))
            process.start()
            
            self.worker_processes.append(process)
        
        # 启动结果处理线程
        self.result_thread = threading.Thread(target=self._process_results)
        self.result_thread.daemon = True
        self.result_thread.start()
        
        print(f"所有 {self.num_gpus} 个GPU FLUX Fill workers 已启动")
    
    def _process_results(self):
        """后台线程处理结果"""
        while not self.stop_event.is_set():
            try:
                result = self.result_queue.get(timeout=1)
                task_id = result['task_id']
                
                event_to_set = None
                with self.pending_tasks_lock:
                    if task_id in self.pending_tasks:
                        # 将结果传递给等待的任务
                        self.pending_tasks[task_id]['result'] = result
                        event_to_set = self.pending_tasks[task_id]['event']
                if event_to_set:
                    event_to_set.set()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"结果处理线程异常: {e}")
                continue
    
    def submit_task(self, image, mask_image, prompt, seed=42, width=1024, height=1024,
                   guidance_scale=30, num_inference_steps=50, max_sequence_length=512, timeout=600):
        """提交填充任务并等待结果"""
        with self.pending_tasks_lock:
            task_id = f"flux_fill_task_{self.task_counter}_{time.time()}"
            self.task_counter += 1
        
        # 将PIL图像转换为字节数据以便跨进程传递
        image_bytes = image_to_bytes(image) if hasattr(image, 'save') else image
        mask_bytes = image_to_bytes(mask_image) if hasattr(mask_image, 'save') else mask_image
        
        task = {
            'task_id': task_id,
            'image_bytes': image_bytes,
            'mask_bytes': mask_bytes,
            'prompt': prompt,
            'seed': seed,
            'width': width,
            'height': height,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_inference_steps,
            'max_sequence_length': max_sequence_length,
        }
        
        # 创建等待事件
        result_event = threading.Event()
        with self.pending_tasks_lock:
            self.pending_tasks[task_id] = {
                'event': result_event,
                'result': None,
                'submitted_time': time.time()
            }
        
        try:
            # 将任务放入队列
            self.task_queue.put(task, timeout=10)
            
            # 等待结果
            if result_event.wait(timeout=timeout):
                with self.pending_tasks_lock:
                    result = self.pending_tasks.get(task_id, {}).get('result')
                    if task_id in self.pending_tasks:
                        del self.pending_tasks[task_id]
                return result if result is not None else {'success': False, 'error': '未知错误'}
            else:
                # 超时
                with self.pending_tasks_lock:
                    if task_id in self.pending_tasks:
                        del self.pending_tasks[task_id]
                return {'success': False, 'error': '任务超时'}
                
        except queue.Full:
            with self.pending_tasks_lock:
                if task_id in self.pending_tasks:
                    del self.pending_tasks[task_id]
            return {'success': False, 'error': '任务队列已满'}
        except Exception as e:
            with self.pending_tasks_lock:
                if task_id in self.pending_tasks:
                    del self.pending_tasks[task_id]
            return {'success': False, 'error': str(e)}
    
    def get_queue_status(self):
        """获取队列状态"""
        with self.pending_tasks_lock:
            pending_count = len(self.pending_tasks)
        return {
            'task_queue_size': self.task_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'pending_tasks': pending_count,
            'active_workers': len(self.worker_processes),
            'total_gpus': self.num_gpus
        }
    
    def stop(self):
        """停止所有workers"""
        print("停止多GPU FLUX Fill管理器...")
        self.stop_event.set()
        
        # 向每个worker发送停止信号
        for _ in range(self.num_gpus):
            try:
                self.task_queue.put(None, timeout=1)
            except queue.Full:
                pass
        
        # 等待所有进程结束
        for process in self.worker_processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        
        print("多GPU FLUX Fill管理器已停止")


# ============== API相关类和函数 ==============

# 全局GPU管理器实例
gpu_manager = None

def initialize_gpu_manager():
    """初始化全局GPU管理器"""
    global gpu_manager
    if gpu_manager is None:
        try:
            if torch.cuda.is_available():
                print(f"检测到 {torch.cuda.device_count()} 个GPU")
            
            gpu_manager = MultiGPUFluxFillManager(
                model_repo_id, 
                num_gpus=NUM_GPUS_TO_USE,
                task_queue_size=TASK_QUEUE_SIZE
            )
            gpu_manager.start_workers()
            print("GPU FLUX Fill管理器初始化成功")
        except Exception as e:
            print(f"GPU FLUX Fill管理器初始化失败: {e}")
            gpu_manager = None

def image_to_base64(image: Image.Image) -> str:
    """将PIL图像转换为base64字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()

def base64_to_image(base64_str: str) -> Image.Image:
    """将base64字符串转换为PIL图像"""
    buffer = io.BytesIO(base64.b64decode(base64_str))
    return Image.open(buffer)

def image_to_bytes(image: Image.Image) -> bytes:
    """将PIL图像转换为字节数据（用于跨进程传递）"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.getvalue()

def bytes_to_image(image_bytes: bytes) -> Image.Image:
    """将字节数据转换为PIL图像"""
    buffer = io.BytesIO(image_bytes)
    return Image.open(buffer)

def process_uploaded_image(file_content: bytes) -> Image.Image:
    """处理上传的图像文件"""
    image = Image.open(io.BytesIO(file_content))
    # 转换为RGB格式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 限制图像大小
    width, height = image.size
    if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
        ratio = min(MAX_IMAGE_SIZE / width, MAX_IMAGE_SIZE / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"图像已调整大小到 {new_width}x{new_height}")
    
    return image


# ============== Pydantic模型 ==============

class FluxImageFillRequest(BaseModel):
    image: Optional[str] = Field(None, description="输入图像(base64编码)，与image_file二选一")
    mask_image: Optional[str] = Field(None, description="Mask图像(base64编码)，与mask_file二选一")
    prompt: str = Field(..., description="填充描述提示词")
    seed: Optional[int] = Field(None, description="随机种子，不提供则随机生成")
    width: Optional[int] = Field(None, description="输出宽度，不提供则使用原图尺寸")
    height: Optional[int] = Field(None, description="输出高度，不提供则使用原图尺寸")
    guidance_scale: Optional[float] = Field(30, ge=1.0, le=50.0, description="引导尺度")
    num_inference_steps: Optional[int] = Field(50, ge=1, le=100, description="推理步数")
    max_sequence_length: Optional[int] = Field(512, ge=128, le=1024, description="最大序列长度")
    enhance_prompt: Optional[bool] = Field(False, description="是否增强提示词")

class FluxImageFillResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    task_id: Optional[str] = Field(None, description="任务ID")
    original_image: Optional[str] = Field(None, description="原始图像(base64编码)")
    mask_image: Optional[str] = Field(None, description="Mask图像(base64编码)")
    filled_image: Optional[str] = Field(None, description="填充后的图像(base64编码)")
    seed: Optional[int] = Field(None, description="使用的随机种子")
    original_prompt: Optional[str] = Field(None, description="原始提示词")
    enhanced_prompt: Optional[str] = Field(None, description="增强后的提示词")
    gpu_id: Optional[int] = Field(None, description="使用的GPU ID")
    processing_time: Optional[float] = Field(None, description="处理时间(秒)")
    error: Optional[str] = Field(None, description="错误信息")

class SystemStatus(BaseModel):
    active_workers: int = Field(..., description="活跃工作进程数")
    task_queue_size: int = Field(..., description="任务队列大小")
    result_queue_size: int = Field(..., description="结果队列大小")
    pending_tasks: int = Field(..., description="待处理任务数")
    total_gpus: int = Field(..., description="总GPU数量")
    system_ready: bool = Field(..., description="系统是否就绪")


# ============== FastAPI应用 ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化GPU管理器
    print("正在启动FLUX Fill图像填充API服务器...")
    initialize_gpu_manager()
    if gpu_manager is None:
        print("警告: GPU管理器初始化失败，某些功能可能不可用")
    else:
        print("FLUX Fill图像填充API服务器启动完成")
    
    yield
    
    # 关闭时清理资源
    print("正在关闭FLUX Fill图像填充API服务器...")
    if gpu_manager:
        gpu_manager.stop()
    print("FLUX Fill图像填充API服务器已关闭")

app = FastAPI(
    title="FLUX.1-Fill-dev 图像填充API服务器",
    description="基于FLUX.1-Fill-dev模型的图像填充API服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """根端点"""
    return {"message": "欢迎使用FLUX.1-Fill-dev 图像填充API服务器", "status": "运行中"}

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "健康",
        "gpu_manager_ready": gpu_manager is not None,
        "timestamp": time.time()
    }

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """获取系统状态"""
    if gpu_manager is None:
        raise HTTPException(status_code=503, detail="GPU管理器未初始化")
    
    status = gpu_manager.get_queue_status()
    return SystemStatus(
        active_workers=status['active_workers'],
        task_queue_size=status['task_queue_size'],
        result_queue_size=status['result_queue_size'],
        pending_tasks=status['pending_tasks'],
        total_gpus=status['total_gpus'],
        system_ready=True
    )

@app.post("/fill", response_model=FluxImageFillResponse)
def fill_image(request: FluxImageFillRequest):
    """填充图像 - 使用base64格式"""
    
    # 检查GPU管理器
    if gpu_manager is None:
        raise HTTPException(status_code=503, detail="GPU管理器未初始化")
    
    if not request.image:
        raise HTTPException(status_code=400, detail="必须提供输入图像")
    
    if not request.mask_image:
        raise HTTPException(status_code=400, detail="必须提供mask图像")
    
    start_time = time.time()
    
    try:
        # 解码输入图像和mask
        input_image = base64_to_image(request.image)
        mask_image = base64_to_image(request.mask_image)
        original_image_b64 = request.image
        mask_image_b64 = request.mask_image
        
        # 获取图像尺寸
        original_width, original_height = input_image.size
        width = request.width or original_width
        height = request.height or original_height
        
        # 处理种子
        if request.seed is None:
            seed = random.randint(0, MAX_SEED)
        else:
            seed = request.seed
        
        # 处理提示词
        original_prompt = request.prompt
        if request.enhance_prompt:
            try:
                enhanced_prompt = rewrite(request.prompt)
                print(f"原始提示词: {original_prompt}")
                print(f"增强提示词: {enhanced_prompt}")
            except Exception as e:
                print(f"提示词增强失败: {e}，使用原始提示词")
                enhanced_prompt = original_prompt
        else:
            enhanced_prompt = original_prompt
        
        # 提交任务到GPU队列
        result = gpu_manager.submit_task(
            image=input_image,
            mask_image=mask_image,
            prompt=enhanced_prompt,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            max_sequence_length=request.max_sequence_length,
            timeout=TASK_TIMEOUT,
        )
        
        if result['success']:
            # 将填充后的图像转换为base64
            filled_image_b64 = image_to_base64(result['image'])
            processing_time = time.time() - start_time
            
            print(f"FLUX图像填充成功，使用GPU {result['gpu_id']}，耗时 {processing_time:.2f} 秒")
            
            return FluxImageFillResponse(
                success=True,
                task_id=result['task_id'],
                original_image=original_image_b64,
                mask_image=mask_image_b64,
                filled_image=filled_image_b64,
                seed=seed,
                original_prompt=original_prompt,
                enhanced_prompt=enhanced_prompt if request.enhance_prompt else None,
                gpu_id=result['gpu_id'],
                processing_time=processing_time
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"FLUX图像填充失败: {result['error']}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"填充图像时发生异常: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/fill-upload", response_model=FluxImageFillResponse)
def fill_image_upload(
    image_file: UploadFile = File(..., description="输入图像文件"),
    mask_file: UploadFile = File(..., description="Mask图像文件"),
    prompt: str = Form(..., description="填充描述提示词"),
    seed: Optional[int] = Form(None, description="随机种子"),
    width: Optional[int] = Form(None, description="输出宽度"),
    height: Optional[int] = Form(None, description="输出高度"),
    guidance_scale: float = Form(30, description="引导尺度"),
    num_inference_steps: int = Form(50, description="推理步数"),
    max_sequence_length: int = Form(512, description="最大序列长度"),
    enhance_prompt: bool = Form(False, description="是否增强提示词")
):
    """填充图像 - 使用文件上传格式"""
    
    # 检查GPU管理器
    if gpu_manager is None:
        raise HTTPException(status_code=503, detail="GPU管理器未初始化")
    
    # 验证文件类型
    if not image_file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="上传的图像文件不是图像格式")
    
    if not mask_file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="上传的mask文件不是图像格式")
    
    start_time = time.time()
    
    try:
        # 读取上传的图像文件
        image_content = image_file.file.read()
        mask_content = mask_file.file.read()
        input_image = process_uploaded_image(image_content)
        mask_image = process_uploaded_image(mask_content)
        
        # 将原始图像转换为base64以便返回
        original_image_b64 = image_to_base64(input_image)
        mask_image_b64 = image_to_base64(mask_image)
        
        # 获取图像尺寸
        original_width, original_height = input_image.size
        width = width or original_width
        height = height or original_height
        
        # 处理种子
        if seed is None:
            seed = random.randint(0, MAX_SEED)
        
        # 处理提示词
        if enhance_prompt:
            try:
                enhanced_prompt = rewrite(prompt)
                print(f"原始提示词: {prompt}")
                print(f"增强提示词: {enhanced_prompt}")
            except Exception as e:
                print(f"提示词增强失败: {e}，使用原始提示词")
                enhanced_prompt = prompt
        else:
            enhanced_prompt = prompt
        
        # 提交任务到GPU队列
        result = gpu_manager.submit_task(
            image=input_image,
            mask_image=mask_image,
            prompt=enhanced_prompt,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            timeout=TASK_TIMEOUT,
        )
        
        if result['success']:
            # 将填充后的图像转换为base64
            filled_image_b64 = image_to_base64(result['image'])
            processing_time = time.time() - start_time
            
            print(f"FLUX图像填充成功，使用GPU {result['gpu_id']}，耗时 {processing_time:.2f} 秒")
            
            return FluxImageFillResponse(
                success=True,
                task_id=result['task_id'],
                original_image=original_image_b64,
                mask_image=mask_image_b64,
                filled_image=filled_image_b64,
                seed=seed,
                original_prompt=prompt,
                enhanced_prompt=enhanced_prompt if enhance_prompt else None,
                gpu_id=result['gpu_id'],
                processing_time=processing_time
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"FLUX图像填充失败: {result['error']}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"填充图像时发生异常: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


# ============== 启动脚本 ==============

def cleanup():
    """清理函数"""
    if gpu_manager:
        gpu_manager.stop()

if __name__ == "__main__":
    def signal_handler(signum, frame):
        print(f"收到信号 {signum}，正在清理资源...")
        cleanup()
        exit(0)
    
    # 注册清理函数
    atexit.register(cleanup)
    
    # 处理信号
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 启动服务器
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8006,  # FLUX Fill端口
            workers=1,  # 必须为1，因为我们使用多进程GPU管理器
            log_level="info"
        )
    except KeyboardInterrupt:
        print("收到中断信号，正在清理资源...")
        cleanup()
    except Exception as e:
        print(f"应用异常: {e}")
        cleanup()
        raise

