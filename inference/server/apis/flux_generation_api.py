#!/usr/bin/env python3
"""
FLUX.1-Krea-dev API服务器
基于Flux模型的GPU管理器，提供REST API接口
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
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image

# 设置多进程启动方法
mp.set_start_method('spawn', force=True)

# 添加src/examples到路径，以便导入tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'examples'))
from tools.prompt_utils import rewrite

from diffusers import FluxPipeline
import torch

# Configuration parameters
# Please replace with your own model path
model_repo_id = "/path/to/FLUX.1-Krea-dev"  # Replace with your FLUX.1-Krea-dev model path

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

NUM_GPUS_TO_USE = int(os.environ.get("NUM_GPUS_TO_USE", torch.cuda.device_count()))  
TASK_QUEUE_SIZE = int(os.environ.get("TASK_QUEUE_SIZE", 100))  
TASK_TIMEOUT = int(os.environ.get("TASK_TIMEOUT", 600))  # Flux模型通常需要更长时间

print(f"配置信息: 使用 {NUM_GPUS_TO_USE} 个GPU，队列大小 {TASK_QUEUE_SIZE}，超时时间 {TASK_TIMEOUT} 秒")

# 默认负面提示词
DEFAULT_NEGATIVE_PROMPT = "blurry, out of focus, low quality, low resolution, pixelated, grainy, oversaturated, undersaturated, amateur photography, phone camera, poor lighting, harsh shadows, overexposed, underexposed, motion blur, distorted proportions, bad anatomy, deformed hands, extra fingers, missing fingers, modern technology, computers, smartphones, casual clothing, messy classroom, fluorescent lighting, digital displays, watermark, text overlay, signature, logo, compression artifacts, noise, color banding, tilted perspective, poor composition"


# ============== GPU管理器类（适配Flux） ==============
class FluxGPUWorker:
    def __init__(self, gpu_id, model_repo_id, task_queue, result_queue, stop_event):
        self.gpu_id = gpu_id
        self.model_repo_id = model_repo_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.device = f"cuda:{gpu_id}"
        self.pipe = None
        
    def initialize_model(self):
        """在指定GPU上初始化Flux模型"""
        try:
            torch.cuda.set_device(self.gpu_id)
            
            if not os.path.exists(self.model_repo_id):
                print(f"错误: 模型路径不存在 {self.model_repo_id}")
                return False
            
            self.pipe = FluxPipeline.from_pretrained(
                self.model_repo_id, 
                torch_dtype=torch.bfloat16,
                local_files_only=True
            )
            self.pipe = self.pipe.to(self.device)  
            print(f"GPU {self.gpu_id} Flux模型初始化成功")
            return True
        except Exception as e:
            print(f"GPU {self.gpu_id} Flux模型初始化失败: {e}")
            return False
    
    def process_task(self, task):
        """处理单个任务"""
        try:
            task_id = task['task_id']
            prompt = task['prompt']
            negative_prompt = task['negative_prompt']
            seed = task['seed']
            width = task['width']
            height = task['height']
            guidance_scale = task['guidance_scale']
            num_inference_steps = task['num_inference_steps']
            max_sequence_length = task['max_sequence_length']
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            with torch.cuda.device(self.gpu_id):
                image = self.pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=max_sequence_length,
                    generator=generator,
                ).images[0]
            
            return {
                'task_id': task_id,
                'image': image,
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
        
        print(f"GPU {self.gpu_id} Flux worker 启动")
        
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
                print(f"GPU {self.gpu_id} Flux worker 异常: {e}")
                continue
        
        print(f"GPU {self.gpu_id} Flux worker 停止")


# 全局GPU worker函数，用于spawn模式
def flux_gpu_worker_process(gpu_id, model_repo_id, task_queue, result_queue, stop_event):
    worker = FluxGPUWorker(gpu_id, model_repo_id, task_queue, result_queue, stop_event)
    worker.run()


class FluxMultiGPUManager:
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
        
        print(f"初始化Flux多GPU管理器，使用 {self.num_gpus} 个GPU，队列大小 {task_queue_size}")
        
    def start_workers(self):
        """启动所有GPU workers"""
        for gpu_id in range(self.num_gpus):
            process = Process(target=flux_gpu_worker_process, 
                            args=(gpu_id, self.model_repo_id, self.task_queue, 
                                  self.result_queue, self.stop_event))
            process.start()
            
            self.worker_processes.append(process)
        
        # 启动结果处理线程
        self.result_thread = threading.Thread(target=self._process_results)
        self.result_thread.daemon = True
        self.result_thread.start()
        
        print(f"所有 {self.num_gpus} 个Flux GPU workers 已启动")
    
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
    
    def submit_task(self, prompt, negative_prompt="", seed=42, width=1024, height=1024, 
                   guidance_scale=4.5, num_inference_steps=50, max_sequence_length=512, timeout=600):
        """提交任务并等待结果"""
        with self.pending_tasks_lock:
            task_id = f"flux_task_{self.task_counter}_{time.time()}"
            self.task_counter += 1
        
        task = {
            'task_id': task_id,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
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
        print("停止Flux多GPU管理器...")
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
        
        print("Flux多GPU管理器已停止")


# ============== API相关类和函数 ==============

# 全局GPU管理器实例
flux_gpu_manager = None

def initialize_flux_gpu_manager():
    """初始化全局Flux GPU管理器"""
    global flux_gpu_manager
    if flux_gpu_manager is None:
        try:
            if torch.cuda.is_available():
                print(f"检测到 {torch.cuda.device_count()} 个GPU")
            
            flux_gpu_manager = FluxMultiGPUManager(
                model_repo_id, 
                num_gpus=NUM_GPUS_TO_USE,
                task_queue_size=TASK_QUEUE_SIZE
            )
            flux_gpu_manager.start_workers()
            print("Flux GPU管理器初始化成功")
        except Exception as e:
            print(f"Flux GPU管理器初始化失败: {e}")
            flux_gpu_manager = None

def get_flux_image_size(aspect_ratio, max_size=1536):
    """获取Flux图片尺寸，保持在合理范围内"""
    if aspect_ratio == "1:1":
        return 1024, 1024
    elif aspect_ratio == "16:9":
        return 1536, 864
    elif aspect_ratio == "9:16":
        return 864, 1536
    elif aspect_ratio == "4:3":
        return 1344, 1008
    elif aspect_ratio == "3:4":
        return 1008, 1344
    elif aspect_ratio == "3:2":
        return 1536, 1024
    elif aspect_ratio == "2:3":
        return 1024, 1536
    else:
        return 1024, 1024

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


# ============== Pydantic模型 ==============

class FluxImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="生成图像的提示词")
    negative_prompt: Optional[str] = Field(DEFAULT_NEGATIVE_PROMPT, description="负面提示词")
    seed: Optional[int] = Field(None, description="随机种子，不提供则随机生成")
    aspect_ratio: Optional[str] = Field("1:1", description="宽高比，选项: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3")
    guidance_scale: Optional[float] = Field(4.5, ge=0.0, le=20.0, description="引导尺度")
    num_inference_steps: Optional[int] = Field(50, ge=1, le=100, description="推理步数")
    max_sequence_length: Optional[int] = Field(512, ge=128, le=1024, description="最大序列长度")
    enhance_prompt: Optional[bool] = Field(True, description="是否增强提示词")

class FluxImageGenerationResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    task_id: Optional[str] = Field(None, description="任务ID")
    image: Optional[str] = Field(None, description="生成的图像(base64编码)")
    seed: Optional[int] = Field(None, description="使用的随机种子")
    enhanced_prompt: Optional[str] = Field(None, description="增强后的提示词")
    gpu_id: Optional[int] = Field(None, description="使用的GPU ID")
    generation_time: Optional[float] = Field(None, description="生成时间(秒)")
    error: Optional[str] = Field(None, description="错误信息")

class FluxSystemStatus(BaseModel):
    active_workers: int = Field(..., description="活跃工作进程数")
    task_queue_size: int = Field(..., description="任务队列大小")
    result_queue_size: int = Field(..., description="结果队列大小")
    pending_tasks: int = Field(..., description="待处理任务数")
    total_gpus: int = Field(..., description="总GPU数量")
    system_ready: bool = Field(..., description="系统是否就绪")


# ============== FastAPI应用 ==============

app = FastAPI(
    title="FLUX.1-Krea-dev API服务器",
    description="基于FLUX.1-Krea-dev模型的图像生成API服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化Flux GPU管理器"""
    print("正在启动Flux API服务器...")
    initialize_flux_gpu_manager()
    if flux_gpu_manager is None:
        print("警告: Flux GPU管理器初始化失败，某些功能可能不可用")
    else:
        print("Flux API服务器启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    print("正在关闭Flux API服务器...")
    if flux_gpu_manager:
        flux_gpu_manager.stop()
    print("Flux API服务器已关闭")

@app.get("/")
async def root():
    """根端点"""
    return {"message": "欢迎使用FLUX.1-Krea-dev API服务器", "status": "运行中"}

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "健康",
        "flux_gpu_manager_ready": flux_gpu_manager is not None,
        "timestamp": time.time()
    }

@app.get("/status", response_model=FluxSystemStatus)
async def get_system_status():
    """获取系统状态"""
    if flux_gpu_manager is None:
        raise HTTPException(status_code=503, detail="Flux GPU管理器未初始化")
    
    status = flux_gpu_manager.get_queue_status()
    return FluxSystemStatus(
        active_workers=status['active_workers'],
        task_queue_size=status['task_queue_size'],
        result_queue_size=status['result_queue_size'],
        pending_tasks=status['pending_tasks'],
        total_gpus=status['total_gpus'],
        system_ready=True
    )

@app.post("/generate", response_model=FluxImageGenerationResponse)
def generate_flux_image(request: FluxImageGenerationRequest):
    """生成Flux图像"""
    
    # 检查GPU管理器
    if flux_gpu_manager is None:
        raise HTTPException(status_code=503, detail="Flux GPU管理器未初始化")
    
    start_time = time.time()
    
    try:
        # 处理种子
        if request.seed is None:
            seed = random.randint(0, MAX_SEED)
        else:
            seed = request.seed
        
        # 获取图像尺寸
        width, height = get_flux_image_size(request.aspect_ratio)
        
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
        result = flux_gpu_manager.submit_task(
            prompt=enhanced_prompt,
            negative_prompt=request.negative_prompt,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            max_sequence_length=request.max_sequence_length,
            timeout=TASK_TIMEOUT,
        )
        
        if result['success']:
            # 将图像转换为base64
            image_b64 = image_to_base64(result['image'])
            generation_time = time.time() - start_time
            
            print(f"Flux图像生成成功，使用GPU {result['gpu_id']}，耗时 {generation_time:.2f} 秒")
            
            return FluxImageGenerationResponse(
                success=True,
                task_id=result['task_id'],
                image=image_b64,
                seed=seed,
                enhanced_prompt=enhanced_prompt if request.enhance_prompt else None,
                gpu_id=result['gpu_id'],
                generation_time=generation_time
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Flux图像生成失败: {result['error']}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"生成Flux图像时发生异常: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


# ============== 启动脚本 ==============

def cleanup():
    """清理函数"""
    if flux_gpu_manager:
        flux_gpu_manager.stop()

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
            port=8002,  # FLUX图像生成端口
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
