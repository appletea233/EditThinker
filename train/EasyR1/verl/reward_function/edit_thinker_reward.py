# -*- coding: utf-8 -*-
# Rewards for multimodal tasks with <think>...</think><answer>...</answer> outputs.
import re
import json
import base64
import io
import os
import random
import copy
from typing import Any, Dict, List, Optional, Tuple
import threading
import time

import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from tqdm import tqdm

from viescore import VIEScore

# ===================== GPT-4.1 Scorer Configuration =====================
# GPT-4.1 API key file path list (supports multiple keys for load balancing)
# Please replace with your own API keys
GPT41_KEY_PATHS: List[str] = [
    "your_api_key_here"  # Replace with your GPT-4.1 API key
]

# Azure Endpoint URL
# Please replace with your own endpoint URL
GPT41_AZURE_ENDPOINT: str = "https://your-endpoint.com/v1/openai/native"

# GPT-4.1 configuration parameters
GPT41_BACKBONE: str = "gpt-4.1"
GPT41_TASK: str = "tie"  # "tie" for image editing evaluation
GPT41_MAX_CLIENT_RETRIES: int = 10
GPT41_MAX_WORKERS: int = 128  # Maximum worker threads for parallel calls

# Whether to enable GPT-4.1 scoring (if False, will use default placeholder score)
GPT41_ENABLED: bool = True

# GPT-4.1 scorer singleton instance
_GPT41_SCORER_INSTANCE: Optional["GPT41EditScorer"] = None
# ==========================================================

# ===================== Image Edit API Configuration =====================
# Edit API endpoint configuration (choose one of two methods)
# Method 1: Directly specify a single endpoint URL
# EDIT_API_ENDPOINT: Optional[str] = None  # Example: "http://your-ip:8080"
EDIT_API_ENDPOINT: Optional[str] = "http://your-ip:8080"  # Replace with your image editing API endpoint
# Method 2: Load endpoint list from file (recommended, supports load balancing)
# File format: one IP address per line (will automatically add http:// and :8080)
# Can be set via environment variable EDIT_API_ENDPOINT_FILE, if not set will use default value
EDIT_API_ENDPOINT_FILE: Optional[str] = os.getenv(
    "EDIT_API_ENDPOINT_FILE",
    None
)  # Read from environment variable EDIT_API_ENDPOINT_FILE, if not set use default path

# Default edit model name to use
EDIT_MODEL_NAME: str = "flux-kontext-dev"  # Options: "omnigen2", "flux-kontext-dev", "qwen-image-edit"

# Edit API maximum retry count
EDIT_API_MAX_RETRIES: int = 10

# Edit API maximum parallel worker threads
EDIT_API_MAX_WORKERS: int = 256


MAX_DURATION: int = 70
# Edit API endpoint pool singleton instance
_EDIT_API_ENDPOINT_POOL: Optional["EndpointPool"] = None
# ==========================================================

# -------------------------
# Patterns for format check
# -------------------------
ANSWER_CAPTURE_PATTERN = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.DOTALL
)

SCORE_CAPTURE_PATTERN = re.compile(
    r"<score>\s*(.*?)\s*</score>",
    re.DOTALL
)


# -------------------------
# Utilities
# -------------------------
def extract_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = ANSWER_CAPTURE_PATTERN.search(text)
    return m.group(1).strip() if m else None


def extract_score(text: str) -> Optional[dict]:
    """Extract JSON content from <score>...</score> tags"""
    if not isinstance(text, str):
        return None
    m = SCORE_CAPTURE_PATTERN.search(text)
    if not m:
        return None
    try:
        score_content = m.group(1).strip()
        score_dict = json.loads(score_content)
        return score_dict if isinstance(score_dict, dict) else None
    except (json.JSONDecodeError, Exception):
        return None


def tag_format_reward(response: str) -> float:
    has_reasoning = bool(re.search(r'<think>.*?</think>', response, re.DOTALL | re.IGNORECASE))
    has_score = bool(re.search(r'<score>.*?</score>', response, re.DOTALL | re.IGNORECASE))
    has_answer = bool(re.search(r'<answer>.*?</answer>', response, re.DOTALL | re.IGNORECASE))

    if not (has_reasoning and has_score and has_answer):
        return 0.0
    
    # Check if the order is correct
    reasoning_match = re.search(r'<think>', response, re.IGNORECASE)
    score_match = re.search(r'<score>', response, re.IGNORECASE)
    answer_match = re.search(r'<answer>', response, re.IGNORECASE)
    
    reasoning_pos = reasoning_match.start() if reasoning_match else -1
    score_pos = score_match.start() if score_match else -1
    answer_pos = answer_match.start() if answer_match else -1
    
    if not (reasoning_pos < score_pos < answer_pos):
        return 0.0

    score_dict = extract_score(response)
    # print(f"score_dict: {score_dict}")
    if not score_dict:
        return 0.0
    
    required_fields = ["semantic", "quality"]
    has_all_required = all(field in score_dict for field in required_fields)
    fields_valid = all(
        isinstance(score_dict.get(field), (int, float)) 
        for field in required_fields 
        if field in score_dict
    )
    
    return 1.0 if (has_all_required and fields_valid) else 0.0

# -------------------------
# Image Edit API Utility Functions
# -------------------------
class EndpointPool:
    """API endpoint pool, supports round-robin access to multiple endpoints"""
    
    def __init__(self, endpoints: List[str], verbose: bool = False):
        """
        Initialize endpoint pool
        
        Args:
            endpoints: List of endpoints, e.g. ["http://ip1:port", "http://ip2:port"]
            verbose: Whether to print initialization information
        """
        if not endpoints:
            raise ValueError("Endpoint list cannot be empty")
        
        self.endpoints = endpoints
        self.current_index = 0
        self.lock = threading.Lock()
        
        if verbose:
            print(f"📡 Initializing Edit API endpoint pool with {len(self.endpoints)} endpoints:")
            for i, endpoint in enumerate(self.endpoints):
                print(f"  [{i+1}] {endpoint}")
    
    def get_next_endpoint(self) -> str:
        """
        Get next endpoint (thread-safe round-robin)
        
        Returns:
            Next endpoint URL
        """
        with self.lock:
            endpoint = self.endpoints[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.endpoints)
            return endpoint


def load_endpoints_from_file(file_path: str, default_port: int = 8007, default_protocol: str = "http") -> List[str]:
    """
    Load endpoint list from file
    
    Args:
        file_path: File path, one IP address per line
        default_port: Default port number
        default_protocol: Default protocol (http or https)
    
    Returns:
        List of endpoint URLs
    """
    endpoints = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # If line is already a complete URL, use it directly
                if line.startswith('http://') or line.startswith('https://'):
                    endpoints.append(line)
                else:
                    # Otherwise add protocol and port
                    endpoints.append(f"{default_protocol}://{line}:{default_port}")
        if not endpoints:
            print(f"Warning: No valid endpoints read from file {file_path}")
    except Exception as e:
        print(f"Error: Failed to read endpoint file {file_path}: {e}")
    return endpoints


def get_edit_api_endpoint_pool() -> Optional[EndpointPool]:
    """
    Get edit API endpoint pool singleton instance
    
    Returns:
        EndpointPool instance, returns None if configuration is not enabled
    """
    global _EDIT_API_ENDPOINT_POOL
    
    if _EDIT_API_ENDPOINT_POOL is not None:
        return _EDIT_API_ENDPOINT_POOL
    
    # Prioritize loading endpoint list from file
    if EDIT_API_ENDPOINT_FILE:
        endpoints = load_endpoints_from_file(EDIT_API_ENDPOINT_FILE)
        if endpoints:
            _EDIT_API_ENDPOINT_POOL = EndpointPool(endpoints, verbose=True)
            return _EDIT_API_ENDPOINT_POOL
    
    # If no file configuration, use single endpoint
    if EDIT_API_ENDPOINT:
        _EDIT_API_ENDPOINT_POOL = EndpointPool([EDIT_API_ENDPOINT], verbose=False)
        return _EDIT_API_ENDPOINT_POOL
    
    print("Warning: Edit API endpoint not configured. Please set EDIT_API_ENDPOINT or EDIT_API_ENDPOINT_FILE")
    return None


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def decode_base64_to_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    img_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(img_data))
    return image


def get_model_config(model_name: str) -> dict:
    """
    Get configuration parameters based on model name
    
    Args:
        model_name: Model name, supports "omnigen2", "flux-kontext-dev", "qwen-image-edit"
    
    Returns:
        Model configuration dictionary
    
    Raises:
        ValueError: If model name is not supported
    """
    if model_name == "omnigen2":
        return {
            'num_inference_step': 50,
            'text_guidance_scale': 5.0,
            'image_guidance_scale': 1.5,
            'width': 1024,
            'height': 1024,
            'negative_prompt': "",
            'enhance_prompt': False,
        }
    elif model_name == "flux-kontext-dev":
        return {
            'num_inference_step': 28,
            'guidance_scale': 2.5,
            'width': 1024,
            'height': 1024,
            'enhance_prompt': False,
        }
    elif model_name == "qwen-image-edit":
        return {
            'num_inference_step': 50,
            'guidance_scale': 5.0,
            'width': 1024,
            'height': 1024,
            'enhance_prompt': False,
        }
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Supported models: 'omnigen2', 'flux-kontext-dev', 'qwen-image-edit'")


def call_edit_model_api(
    instruction: str,
    input_image: Image.Image,
    endpoint_pool: Optional[EndpointPool] = None,
    model_config: dict = None,
    max_retries: int = 3,
    total_timeout: Optional[float] = None,
    start_time: Optional[float] = None
) -> Optional[Image.Image]:
    """
    Call image edit model API to generate edited image
    
    Args:
        instruction: Edit instruction
        input_image: Input image
        endpoint_pool: API endpoint pool (if None, will use global configuration)
        model_config: Model configuration parameters
        max_retries: Maximum retry count
        total_timeout: Total timeout (seconds), if start_time is set, will check before each retry
        start_time: Start timestamp, used to calculate elapsed time
    
    Returns:
        Edited PIL Image, returns None if failed
    """
    
    image_b64 = encode_image_to_base64(input_image)
    
    request_data = {
        "image": image_b64,
        "prompt": instruction,
        **(model_config or {}),
    }
    
    total_attempts = max_retries * len(endpoint_pool.endpoints)
    current_endpoint = None
    for attempt in range(total_attempts):
        # If total timeout is set, check if already timed out
        if total_timeout is not None and start_time is not None:
            elapsed = time.time() - start_time
            if elapsed >= total_timeout:
                return None
            # Calculate timeout for this request (not exceeding total timeout)
            remaining_timeout = max(1.0, total_timeout - elapsed)
        else:
            remaining_timeout = 50  # Default 50 seconds
        
        try:
            current_endpoint = endpoint_pool.get_next_endpoint()
            
            # Set shorter timeout (50 seconds or remaining time), leave buffer for outer 60-second timeout
            response = requests.post(
                f"{current_endpoint}/edit",
                json=request_data,
                timeout=min(remaining_timeout, 65)
            )
            
            result = response.json()
            
            if result.get("success"):
                output_image = decode_base64_to_image(result["edited_image"])
                return output_image
            else:
                print(f"Warning: Edit API returned error (endpoint: {current_endpoint}): {result.get('error', 'Unknown error')}")
                if attempt < total_attempts - 1:
                    continue
                    
        except Exception as e:
            endpoint_str = current_endpoint if current_endpoint else "unknown"
            print(f"Warning: Edit API call failed (attempt {attempt + 1}/{total_attempts}, endpoint: {endpoint_str}): {e}")
            if attempt < total_attempts - 1:
                continue
    
    return None


class GPT41EditScorer:
    """Wrapper for GPT-4.1 (VIEScore) based image edit evaluation."""

    def __init__(
        self,
        key_paths: List[str],
        azure_endpoint: str,
        backbone: str = "gpt-4.1",
        task: str = "tie",
        max_client_retries: int = 3,
    ) -> None:
        if VIEScore is None:
            raise ImportError(
                "VIEScore is not available. Please ensure viescore is importable "
                "and required dependencies are installed."
            )

        if not key_paths:
            raise ValueError("key_paths must be provided to initialise GPT-4.1 scorer.")

        self._clients = [
            VIEScore(
                backbone=backbone,
                task=task,
                key_path=key_path,
                azure_endpoint=azure_endpoint,
            )
            for key_path in key_paths
        ]
        self._lock = threading.Lock()
        self._next_index = 0
        self._max_client_retries = max(1, max_client_retries)

    def _get_next_client(self) -> "VIEScore":
        with self._lock:
            client = self._clients[self._next_index % len(self._clients)]
            self._next_index += 1
        return client

    def score(
        self,
        original_image: Image.Image,
        edited_image: Image.Image,
        instruction: str,
        resize_to_match: bool = True,
        fallback_score: float = 0.0,
    ) -> Tuple[float, Dict[str, Any]]:
        if not self._clients:
            return fallback_score, {"error": "no_available_clients"}

        prompt = (instruction or "").strip()
        if not prompt:
            return fallback_score, {"error": "empty_instruction"}

        original_rgb = original_image.convert("RGB")
        edited_rgb = edited_image.convert("RGB")
        if resize_to_match and edited_rgb.size != original_rgb.size:
            edited_rgb = edited_rgb.resize(original_rgb.size)

        total_attempts = len(self._clients) * self._max_client_retries
        attempts = 0
        last_error: Optional[Exception] = None

        while attempts < total_attempts:
            attempts += 1
            client = self._get_next_client()
            try:
                semantics_score, quality_score, overall_score = client.evaluate(
                    [original_rgb, edited_rgb], prompt
                )
                return overall_score, {
                    "semantic": semantics_score,
                    "quality": quality_score,
                    "overall": overall_score,
                }
            except Exception as exc:  # pragma: no cover - network/third-party failure
                last_error = exc
                print(f"Warning: GPT-4.1 evaluation failed (attempt {attempts}/{total_attempts}): {exc}")
                time.sleep(1)

        error_msg = str(last_error) if last_error else "unknown_error"
        print(f"Error: GPT-4.1 evaluation exhausted retries: {error_msg}")
        return fallback_score, {"error": error_msg}


def get_gpt41_scorer() -> Optional[GPT41EditScorer]:
    """
    Get GPT-4.1 scorer singleton instance (using internal file configuration)
    
    Returns:
        GPT41EditScorer instance, returns None if configuration is not enabled or VIEScore is unavailable
    """
    global _GPT41_SCORER_INSTANCE
    
    if not GPT41_ENABLED:
        return None
    
    if VIEScore is None:
        print("Warning: VIEScore is not available. GPT-4.1 scorer will not be used.")
        return None
    
    if not GPT41_KEY_PATHS or not GPT41_AZURE_ENDPOINT:
        print("Warning: GPT-4.1 configuration is incomplete. Please set GPT41_KEY_PATHS and GPT41_AZURE_ENDPOINT in edit_thinker_reward.py")
        return None
    
    if _GPT41_SCORER_INSTANCE is None:
        try:
            _GPT41_SCORER_INSTANCE = GPT41EditScorer(
                key_paths=GPT41_KEY_PATHS,
                azure_endpoint=GPT41_AZURE_ENDPOINT,
                backbone=GPT41_BACKBONE,
                task=GPT41_TASK,
                max_client_retries=GPT41_MAX_CLIENT_RETRIES,
            )
            print(f"GPT-4.1 scorer initialized with {len(GPT41_KEY_PATHS)} key(s) and endpoint: {GPT41_AZURE_ENDPOINT}")
        except Exception as e:
            print(f"Error: Failed to initialize GPT-4.1 scorer: {e}")
            _GPT41_SCORER_INSTANCE = None
    
    return _GPT41_SCORER_INSTANCE


def evaluate_image_prompt_alignment_gpt41(
    original_image: Optional[Image.Image],
    edited_image: Optional[Image.Image],
    instruction: str,
    scorer: Optional[GPT41EditScorer] = None,
    fallback_score: float = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate image-instruction alignment using GPT-4.1
    
    Args:
        original_image: Original image
        edited_image: Edited image
        instruction: Edit instruction
        scorer: GPT-4.1 scorer instance (if None, will use internal file configuration)
        fallback_score: Default score when failed
    
    Returns:
        (score, details dictionary)
    """
    if original_image is None or edited_image is None:
        return fallback_score, {"error": "missing_image"}

    if scorer is None:
        scorer = get_gpt41_scorer()
        if scorer is None:
            return fallback_score, {"error": "scorer_not_available"}

    return scorer.score(original_image, edited_image, instruction, fallback_score=fallback_score)


def accuracy_reward(response: str,
                    ground_truth: dict) -> float:
    try:
        gt_scores = ground_truth or ""
        predicted_scores = extract_score(response)

        print("accuracy_score", predicted_scores, "gt_scores", gt_scores)
        
        score_fields = ["semantic", "quality"]
        squared_errors = []
        valid_fields = 0
        
        for field in score_fields:
            gt_field = "semantics" if field == "semantic" else field
            if field in predicted_scores and gt_field in gt_scores:
                pred_val = float(predicted_scores[field])
                gt_val = float(gt_scores[gt_field])
                error = pred_val - gt_val
                squared_errors.append(error * error)
                valid_fields += 1
        mse = sum(squared_errors) / valid_fields
        reward = max(0.0, 10.0 - mse)
        return reward / 10.0
    except Exception as e:
        print(f"Error in accuracy_reward: {e}")
        return 0.0


def get_edited_image_and_score(edit_image_queue, endpoint_pool, edit_model_config, gpt41_scorer):

    def call_single_internal(info):
        if info["skip_image_edit"]:
            return info["idx"], None

        """Internal function to perform actual image editing and scoring"""
        start_time = time.time()
        
        instruction = info["refined_prompt"]
        image_path = info["original_image_path"]
        image = Image.open(image_path).convert("RGB")
        
        # Check if already timed out
        if time.time() - start_time >= MAX_DURATION:
            return info["idx"], None
        
        # Override default config with item-specific seed
        item_model_config = copy.deepcopy(edit_model_config)
        if "seed" in info:
            item_model_config["seed"] = info["seed"]
        
        edited_image = call_edit_model_api(
            instruction=instruction,
            input_image=image,
            endpoint_pool=endpoint_pool,
            model_config=item_model_config,
            max_retries=EDIT_API_MAX_RETRIES,
            total_timeout=MAX_DURATION,
            start_time=start_time
        )

        # Check if already timed out
        if time.time() - start_time >= MAX_DURATION:
            return info["idx"], None

        # If edit failed, return None
        if edited_image is None:
            return info["idx"], None

        score, details = evaluate_image_prompt_alignment_gpt41(
            original_image=image,
            edited_image=edited_image,
            instruction=info["origin_prompt"],
            scorer=gpt41_scorer,
            fallback_score=0.0
        )

        # Check if already timed out
        if time.time() - start_time >= MAX_DURATION:
            return info["idx"], None

        info["gpt_score"] = score
        info["gpt_detail"] = details
        return info["idx"], info

    def call_single(info):
        """Wrapper function to ensure call_single_internal completes within timeout"""
        single_executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = single_executor.submit(call_single_internal, info)
            try:
                result = future.result(timeout=MAX_DURATION)
                return result
            except FutureTimeoutError:
                # Try to cancel task on timeout
                future.cancel()
                print(f"Func get_edited_image_and_score() Error: Sample {info['idx']} timed out ({MAX_DURATION} seconds)")
                return info["idx"], None
            except Exception as e:
                future.cancel()
                print(f"Func get_edited_image_and_score() Error: Sample {info['idx']} processing exception: {e}")
                return info["idx"], None
            finally:
                # Force shutdown executor, interrupt all unfinished tasks
                single_executor.shutdown(wait=False)
        except Exception as e:
            print(f"Func get_edited_image_and_score() Error: Sample {info['idx']} executor exception: {e}")
            return info["idx"], None

    with ThreadPoolExecutor(max_workers=GPT41_MAX_WORKERS) as executor:
        futures = {
            executor.submit(call_single, info): info["idx"]
            for info in edit_image_queue
        }
        with tqdm(total=len(edit_image_queue), desc="Processing image editing and scoring", unit="sample") as pbar:
            for future in as_completed(futures):
                try:
                    idx, info = future.result()
                    edit_image_queue[idx] = info
                    pbar.update(1)
                except Exception as e:
                    idx = futures[future]
                    print(f"Func get_edited_image_and_score() Error: Sample {idx} processing exception: {e}")
                    edit_image_queue[idx] = None
                    pbar.update(1)

    return edit_image_queue

def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.5,
) -> List[Dict[str, float]]:
    results = []
    image_edit_queue = [] 
    for idx, item in enumerate(reward_inputs):
        try:
            raw_response = item["response"]
            response = re.sub(r"\s*(<|>|/)\s*", r"\1", raw_response)
            gt_extracted = item["ground_truth"]

            format_reward = tag_format_reward(response)
            viescore_accuracy_reward = accuracy_reward(response, gt_extracted)
            
            refined_prompt = extract_answer(response) or ""

            # print("refined_prompt", refined_prompt)
            # print("origin_prompt", item["origin_prompt"])
            # print("gt_extracted", gt_extracted)

            
            origin_prompt = item["origin_prompt"]
            previous_image_score = (gt_extracted["semantics"] + gt_extracted["quality"]) ** 0.5
            origin_image_path = item.get("origin_image_path", None)
            
            # Use same seed for every 8 items
            if idx % 8 == 0:
                # seed = random.randint(0, 100000)
                seed = 0

            skip_image_edit = previous_image_score >= 8.0
              
            image_edit_queue.append({
                "idx": idx,
                "original_image_path": origin_image_path,
                "origin_prompt": origin_prompt, 
                "refined_prompt": refined_prompt,
                "viescore_accuracy_reward": viescore_accuracy_reward, 
                "previous_image_score": previous_image_score,
                "previous_image_semantic_score": gt_extracted["semantics"],
                "previous_image_quality_score": gt_extracted["quality"],
                "seed": seed,
                "skip_image_edit": skip_image_edit,
            })


            results.append({
                "overall": 0.0, 
                "format_reward": float(format_reward), 
                "viescore_accuracy_reward": viescore_accuracy_reward, 
                "edited_image_reward_semantic": 0.0,
                "edited_image_reward_quality": 0.0,
                "idx": idx,
            })
        except Exception as e:
            print(f"Func compute_score() Error: Sample {idx} processing exception: {e}")
            results.append({"overall": 0.0, "format_reward": 0.0, "viescore_accuracy_reward": 0.0, "edited_image_reward": 0.0})

    # Get edited images and scores
    edit_endpoint_pool = get_edit_api_endpoint_pool()
    edit_model_config = get_model_config(EDIT_MODEL_NAME)
    gpt41_scorer = get_gpt41_scorer()
            
    image_edit_results = get_edited_image_and_score(image_edit_queue, edit_endpoint_pool, edit_model_config, gpt41_scorer)

    for idx, result in enumerate(image_edit_results):

        if result is None:
            results[idx]['edited_image_reward_semantic'] = 0.5
            results[idx]['edited_image_reward_quality'] = 0.5
            results[idx]['overall'] = 0.5 * results[idx]['format_reward'] + 0.5 * (0.2 * results[idx]['viescore_accuracy_reward'] + 0.6 * results[idx]['edited_image_reward_semantic'] + 0.2 * results[idx]['edited_image_reward_quality'])
            continue

        print(result['gpt_score'], result['gpt_detail'])
        
        edited_image_semantic_score = result['gpt_detail']['semantic']
        edited_image_quality_score = result['gpt_detail']['quality']
        
        edited_image_reward_semantic = edited_image_semantic_score - result['previous_image_semantic_score']
        edited_image_reward_quality = edited_image_quality_score - result['previous_image_quality_score']
        
        edited_image_reward_semantic = edited_image_reward_semantic / 10.0
        edited_image_reward_semantic = (edited_image_reward_semantic + 1.0) / 2.0
        edited_image_reward_quality = edited_image_reward_quality / 10.0
        edited_image_reward_quality = (edited_image_reward_quality + 1.0) / 2.0
        
        results[idx]['edited_image_reward_semantic'] = edited_image_reward_semantic
        results[idx]['edited_image_reward_quality'] = edited_image_reward_quality
        # results[idx]['overall'] = 0.5 * results[idx]['format_reward'] + 0.5 * (0.2 * results[idx]['viescore_accuracy_reward'] + 0.6 * edited_image_reward_semantic + 0.2 * edited_image_reward_quality)
        results[idx]['overall'] = 0.5 * results[idx]['format_reward'] + 0.5 * edited_image_reward_semantic 

    return results
