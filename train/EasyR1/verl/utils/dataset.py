# 

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF
from qwen_vl_utils.vision_process import fetch_video



def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    # print(max_pixels)

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        # print("minminmin!!!!!!!!!!!!")
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str, min_pixels: int = 4*32*32, max_pixels: int = 64*32*32, max_frames: int = 128, video_fps: float = 2, return_fps: bool = False
):
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "max_frames": max_frames, "fps": video_fps}
    return fetch_video(vision_info, image_patch_size=16, return_video_sample_fps=return_fps, return_video_metadata=return_fps)



class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
            desc="Filtering overlong prompts",
            num_proc=filter_overlong_prompts_workers,
        )

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        """Build messages for image-edit task with two images."""
        if not self.format_prompt:
            raise ValueError("format_prompt is not set")
        
        # Replace placeholders in prompt template
        prompt_str = self.format_prompt.strip().replace(
            '{original_instruction}', example.get("origin_prompt", "")
        ).replace(
            '{rewritten_prompt}', example.get("rewritten_prompt", "")
        )
        
        # Build message content with two images
        content_list = []
        parts = prompt_str.split("<image>")
        for i, content in enumerate(parts):
            if i != 0:
                content_list.append({"type": "image"})
            if content:
                content_list.append({"type": "text", "text": content})
        
        return [{"role": "user", "content": content_list}]
       


    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        """Filter out prompts that are too long for image-edit tasks."""
        messages = self._build_messages(example)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # Add image directory prefix if needed
        if self.image_dir is not None:
            if not os.path.isabs(example['origin_image_path']):
                example['origin_image_path'] = os.path.join(self.image_dir, example['origin_image_path'])

            if not os.path.isabs(example['edited_image_path']):
                example['edited_image_path'] = os.path.join(self.image_dir, example['edited_image_path'])

        # Get image paths
        origin_image_path = example.get("origin_image_path")
        edited_image_path = example.get("edited_image_path")
        
        if not origin_image_path or not edited_image_path:
            return False
        
        images = [origin_image_path, edited_image_path]
        
        # Process images
        processed_images = []
        for image in images:
            processed_images.append(process_image(image, self.min_pixels, self.max_pixels))
        
        # Check prompt length
        model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
        return model_inputs["input_ids"].size(-1) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """Get item for image-edit task only."""
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example["problem_id"] = index
        
        # Build prompt
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        
        # Get image paths

        if self.image_dir is not None:
            if not os.path.isabs(example['origin_image_path']):
                example['origin_image_path'] = os.path.join(self.image_dir, example['origin_image_path'])

            if not os.path.isabs(example['edited_image_path']):
                example['edited_image_path'] = os.path.join(self.image_dir, example['edited_image_path'])
        origin_image_path = example.get("origin_image_path")
        edited_image_path = example.get("edited_image_path")
        
        if origin_image_path is None or edited_image_path is None:
            print(example)
            raise ValueError("origin_image_path or edited_image_path is not set")
        
        images = [origin_image_path, edited_image_path]
        
        # Process images
        processed_images = []
        for image in images:
            processed_images.append(process_image(image, self.min_pixels, self.max_pixels))
        
        # Process with model processor
        try:
            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
        except Exception as e:
            print(f"Error processing: {e}")
            print(f"Images: {processed_images}")
            print(f"Prompt: {prompt}")
            print(f"Example: {example}")
            raise e
        
        # Extract model inputs
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        
        # Set metadata
        example["multi_modal_data"] = {"images": images}
        example["data_type"] = 'image-edit'
        example["problem_type"] = 'image-edit'
        
        # Generate position IDs
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # Qwen-VL mRoPE (multi-dimensional rope for vision)
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index
            
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)
        
        # Post-process data (padding, truncation)
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        
        # Process raw prompt IDs
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
        
        # Finalize example
        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        
        return example
