import dotenv

dotenv.load_dotenv(override=True)

import argparse
import os
import json
from tqdm import tqdm
from typing import List, Tuple, Optional
from PIL import Image, ImageOps
import base64
import io
import requests
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import threading

from datasets import load_dataset


class EndpointPool:
    """API endpoint pool with round-robin access to multiple endpoints"""
    
    def __init__(self, endpoints: List[str], verbose: bool = False):
        """
        Initialize endpoint pool
        
        Args:
            endpoints: List of endpoints, e.g. ["http://ip1:port", "http://ip2:port"]
            verbose: Whether to print initialization info
        """
        if not endpoints:
            raise ValueError("Endpoint list cannot be empty")
        
        self.endpoints = endpoints
        self.current_index = 0
        self.lock = threading.Lock()
        
        if verbose:
            print(f"📡 Initializing API endpoint pool with {len(self.endpoints)} endpoints:")
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


class OpenAIClient:
    """OpenAI API client that sends HTTP requests directly using requests"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
    
    def chat_completions_create(self, model: str, messages: list, timeout: int = 180, temperature: Optional[float] = None) -> dict:
        """
        Call OpenAI Chat Completions API
        
        Args:
            model: Model name
            messages: Message list
            timeout: Request timeout in seconds
            temperature: Temperature parameter (optional)
        
        Returns:
            API response dictionary
        """
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
        }
        
        # Add temperature to payload if provided
        if temperature is not None:
            payload["temperature"] = temperature

        if 'gemini' in model:
            payload['extra_body'] = {
                "google": {
                    "thinking_config": {
                        "include_thoughts": True,
                        "thinking_budget": -1
                    },
                    "thought_tag_marker": "think"
                },
            }
            payload['max_tokens'] = 65536
        elif 'Qwen' in model or 'qwen' in model:
            # vLLM Qwen model configuration
            payload['max_tokens'] = 4096
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            if hasattr(response, 'text'):
                print(f"Response content: {response.text}")
            raise


def load_prompt_from_file(file_path: str) -> str:
    """
    Load prompt from file
    
    Args:
        file_path: Prompt file path
    
    Returns:
        str: Prompt content
    """
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None


def polish_edit_prompt(prompt, img, openai_client, model, prompt_template=None):
    """
    Polish and enhance edit instruction prompt using GPT vision model.
    
    Args:
        prompt: Original user instruction
        img: Path to the image to be edited
        openai_client: OpenAIClient instance for API calls
        model: Model name to use
    
    Returns:
        str: Polished prompt
    """
    # Use prompt template loaded from file, or default if not provided
    if prompt_template is None:
        # Default prompt (for backward compatibility)
        EDIT_SYSTEM_PROMPT = '''
# Edit Instruction Rewriter
You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction and the image to be edited.  

Please strictly follow the rewriting rules below:

## 1. General Principles
- Keep the rewritten prompt **concise**. Avoid overly long sentences and reduce unnecessary descriptive language.  
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.  
- Keep the core intention of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.  
- All added objects or modifications must align with the logic and style of the edited input image's overall scene.  

## 2. Task Type Handling Rules
### 1. Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.  
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:  
    > Original: "Add an animal"  
    > Rewritten: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"  
- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.  
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.  

### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes `" "`. Do not translate or alter the original language of the text, and do not change the capitalization.  
- **For text replacement tasks, always use the fixed template:**
    - `Replace "xx" to "yy"`.  
    - `Replace the xx bounding box to "yy"`.  
- If the user does not specify text content, infer and add concise text based on the instruction and the input image's context. For example:  
    > Original: "Add a line of text" (poster)  
    > Rewritten: "Add text \"LIMITED EDITION\" at the top center with slight shadow"  
- Specify text position, color, and layout in a concise way.  

### 3. Human Editing Tasks
- Maintain the person's core visual consistency (ethnicity, gender, age, hairstyle, expression, outfit, etc.).  
- If modifying appearance (e.g., clothes, hairstyle), ensure the new element is consistent with the original style.  
- **For expression changes, they must be natural and subtle, never exaggerated.**  
- If deletion is not specifically emphasized, the most important subject in the original image (e.g., a person, an animal) should be preserved.
    - For background change tasks, emphasize maintaining subject consistency at first.  
- Example:  
    > Original: "Change the person's hat"  
    > Rewritten: "Replace the man's hat with a dark brown beret; keep smile, short hair, and gray jacket unchanged"  

### 4. Style Transformation or Enhancement Tasks
- If a style is specified, describe it concisely with key visual traits. For example:  
    > Original: "Disco style"  
    > Rewritten: "1970s disco: flashing lights, disco ball, mirrored walls, colorful tones"  
- If the instruction says "use reference style" or "keep current style," analyze the input image, extract main features (color, composition, texture, lighting, art style), and integrate them into the prompt.  
- **For coloring tasks, including restoring old photos, always use the fixed template:** "Restore old photograph, remove scratches, reduce noise, enhance details, high resolution, realistic, natural skin tones, clear facial features, no distortion, vintage photo restoration"  
- If there are other changes, place the style description at the end.

## 3. Rationality and Logic Checks
- Resolve contradictory instructions: e.g., "Remove all trees but keep all trees" should be logically corrected.  
- Add missing key information: if position is unspecified, choose a reasonable area based on composition (near subject, empty space, center/edges).  

# Output Format Example
```json
{
   "Rewritten": "..."
}
```
'''
    else:
        EDIT_SYSTEM_PROMPT = prompt_template
    
    prompt_text = f"{EDIT_SYSTEM_PROMPT}\n\nUser Input: {prompt}\n\nRewritten Prompt:"
    
    max_retries = 10
    for attempt in range(max_retries):
        try:
            # Build message content
            content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(img)}"}}
            ]
            
            # Use OpenAIClient to send request
            response = openai_client.chat_completions_create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": content
                }],
                timeout=3600,  # vLLM multimodal tasks may require longer timeout
                # temperature=0.01,
            )
            
            result = response['choices'][0]['message']['content']
            
            if isinstance(result, str):
                result = result.replace('```json','')
                result = result.replace('```','')
                result = json.loads(result)
            else:
                result = json.loads(result)

            polished_prompt = result['Rewritten']
            polished_prompt = polished_prompt.strip()
            polished_prompt = polished_prompt.replace("\n", " ")
            return polished_prompt
            
        except Exception as e:
            print(f"[Warning] Error during polish_edit_prompt API call (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(f"[Error] Failed to polish prompt after {max_retries} attempts, returning original prompt")
                return prompt


def evaluate_and_rewrite_prompt(
    original_image_path, 
    original_prompt, 
    rewritten_prompt, 
    edited_image_path, 
    openai_client, 
    model,
    prompt_template=None
):
    """
    Evaluate if the edited image matches the original prompt intent.
    If not, rewrite the prompt based on all available information.
    
    Args:
        original_image_path: Path to the original input image
        original_prompt: The original user instruction
        rewritten_prompt: The previously rewritten prompt that was used
        edited_image_path: Path to the edited image generated from rewritten_prompt
        openai_client: OpenAI client for API calls
        model: Model name to use
    
    Returns:
        dict: {
            'is_satisfied': bool,  # Whether the edit satisfies the original intent
            'reason': str,  # Explanation of the evaluation
            'new_rewritten_prompt': str or None  # New rewritten prompt if refinement needed
        }
    """
    # Use prompt template loaded from file, or default if not provided
    if prompt_template is None:
        # Default prompt (for backward compatibility)
        EVALUATE_REWRITE_PROMPT = '''
# Edit Evaluation and Prompt Refinement System

You are an expert image editing evaluator and prompt engineer. Your task is to:
1. Evaluate whether an edited image successfully fulfills the original user instruction
2. If not satisfied, generate an improved rewritten prompt that addresses the shortcomings

## Input Information
You will receive:
- **Original Image**: The input image before editing
- **Original User Instruction**: The user's initial editing request
- **Rewritten Prompt**: The refined instruction that was used for editing
- **Edited Image**: The resulting image after applying the rewritten prompt

## Evaluation Criteria

### A. Intent Alignment
- Does the edited image achieve the core goal of the original instruction?
- Are all requested changes present and correctly implemented?

### B. Quality Assessment
- **Subject/Object Changes**: Are additions/removals/replacements accurate?
- **Appearance Modifications**: Are color/material/style changes applied correctly?
- **Scene Changes**: Is background/environment modification satisfactory?
- **Detail Preservation**: Are important details maintained where needed?
- **Visual Coherence**: Does the edit look natural and well-integrated?

### C. Common Failure Patterns to Check
- Missing requested elements
- Incorrect positioning or scale
- Wrong colors or materials
- Unnatural blending or artifacts
- Loss of important subject details
- Style inconsistency
- Text errors (if applicable)
- Over-editing or under-editing

## Evaluation Decision

**SATISFIED**: The edited image successfully fulfills the original instruction with acceptable quality.
- Minor imperfections are acceptable if the core intent is met
- Use this when the edit is "good enough" for the user's purpose

**NOT SATISFIED**: The edited image fails to meet the original instruction in significant ways.
- Major elements are missing or incorrect
- Quality issues severely impact the result
- The rewritten prompt needs refinement

## Prompt Refinement Strategy (if NOT SATISFIED)

When generating a new rewritten prompt, analyze:

1. **What went wrong?**
   - Compare original instruction → rewritten prompt → edited result
   - Identify gaps between intent and execution
   - Determine if the issue is clarity, specificity, or contradiction

2. **Refinement Approaches:**

   **If the rewritten prompt was too vague:**
   - Add more specific descriptors (exact colors, positions, sizes)
   - Include spatial relationships and context
   - Specify interaction with existing elements
   
   **If the rewritten prompt was contradictory:**
   - Resolve conflicts between requirements
   - Prioritize core intent over secondary details
   - Simplify complex multi-part instructions
   
   **If important details were lost:**
   - Explicitly state preservation requirements
   - Add "maintain [aspect]" or "preserve [feature]" clauses
   - Reference specific elements from the original image
   
   **If positioning/scale was wrong:**
   - Use more precise spatial descriptors
   - Add relative size/scale indicators
   - Specify foreground/midground/background placement
   
   **If style/appearance was incorrect:**
   - Use more specific visual vocabulary
   - Add reference to original image's style elements
   - Include material/texture/lighting specifications
   
   **If the edit was over/under-processed:**
   - Add modifiers like "subtle", "gentle", "dramatic", "significant"
   - Specify degree of change more clearly
   - Balance enhancement with naturalness

3. **Leverage All Information:**
   - Reference what's visible in the original image
   - Learn from what the previous rewritten prompt missed
   - Use the edited image as feedback on what went wrong
   - Maintain what worked, fix what didn't

## Output Format

```json
{
    "is_satisfied": true/false,
    "reason": "Detailed explanation of evaluation. If satisfied, explain why it meets requirements. If not satisfied, explain specific shortcomings.",
    "new_rewritten_prompt": "Only include if is_satisfied is false. The improved rewritten prompt that addresses the identified issues. If is_satisfied is true, set this to null."
}
```

## Examples

### Example 1: Satisfied
```json
{
    "is_satisfied": true,
    "reason": "The edited image successfully adds a cat in the bottom-right corner as requested. The cat is appropriately sized, naturally lit, and well-integrated into the scene. Minor shadow artifacts are present but do not detract from the overall quality.",
    "new_rewritten_prompt": null
}
```

### Example 2: Not Satisfied - Need More Specificity
Original: "Change the color"
Rewritten: "Change the object color to blue"
Issue: Wrong object was recolored

```json
{
    "is_satisfied": false,
    "reason": "The rewritten prompt was too vague about which object to recolor. The background was changed to blue instead of the intended subject (the car). The prompt needs to explicitly specify the target object.",
    "new_rewritten_prompt": "Change the car color to blue, maintaining the metallic finish and reflections, keep all other elements including background unchanged"
}
```

### Example 3: Not Satisfied - Lost Important Details
Original: "Change background to beach"
Rewritten: "Replace background with beach scene"
Issue: Subject was altered/degraded

```json
{
    "is_satisfied": false,
    "reason": "While the background was changed to a beach scene, the subject (person) lost facial detail and edge definition became blurry. The rewritten prompt failed to emphasize subject preservation.",
    "new_rewritten_prompt": "Replace background with sunny beach scene featuring sand, ocean, and clear sky, while strictly maintaining the subject's sharpness, facial features, hair details, and edge definition from the original image"
}
```

### Example 4: Not Satisfied - Positioning Error
Original: "Add a lamp"
Rewritten: "Add a lamp"
Issue: Lamp was added in awkward position

```json
{
    "is_satisfied": false,
    "reason": "A lamp was added but placed in the center of the floor, which looks unnatural. The rewritten prompt lacked spatial guidance.",
    "new_rewritten_prompt": "Add a modern floor lamp in the left corner near the sofa, approximately 5-6 feet tall, with warm lighting that complements the room's ambiance"
}
```

Now evaluate the provided images and prompts, and return your analysis in the specified JSON format.
'''
        use_simplified_format = False
    else:
        EVALUATE_REWRITE_PROMPT = prompt_template
        EVALUATE_REWRITE_PROMPT = EVALUATE_REWRITE_PROMPT.replace('<image>', '')
        # Check if using simplified_eval_prompt_v2.txt format (with placeholders)
        use_simplified_format = '{original_instruction}' in prompt_template or '{rewritten_prompt}' in prompt_template
    
    success = False
    max_retries = 10
    for _ in range(max_retries):
        try:
            # If using simplified format, replace placeholders
            if use_simplified_format:
                # Use replace method to substitute placeholders, avoiding JSON braces being mistaken for placeholders
                prompt_text = EVALUATE_REWRITE_PROMPT.replace('{original_instruction}', original_prompt).replace('{rewritten_prompt}', rewritten_prompt)
                content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(original_image_path)}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(edited_image_path)}"}},
                    {"type": "text", "text": prompt_text},
                ]
            else:
                # Default format
                content = [
                    {"type": "text", "text": EVALUATE_REWRITE_PROMPT},
                    {"type": "text", "text": f"\n\n## Input Data:\n\n**Original User Instruction:** {original_prompt}\n\n**Rewritten Prompt Used:** {rewritten_prompt}\n\n**Images Order:** [Original Image, Edited Image]\n\nPlease evaluate:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(original_image_path, 1024)}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(edited_image_path, 1024)}"}},
                ]
            
            # Use OpenAIClient to send request
            response = openai_client.chat_completions_create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": content
                }],
                timeout=360,  # vLLM multimodal tasks may require longer timeout
                # temperature=0.01,
            )
            
            result = response['choices'][0]['message']['content']

            # breakpoint()
            # print(result)
            
            # Parse response
            if use_simplified_format:
                # Parse simplified_eval_prompt_v2.txt format output
                # Format: <think>...</think><score>...</score><answer>...</answer>
                import re
                
                # Extract reason (from <think> tag)
                reason_match = re.search(r'<think>(.*?)</think>', result, re.DOTALL)
                reason = reason_match.group(1).strip() if reason_match else ""
                
                # Extract score (from <score> tag)
                score_match = re.search(r'<score>(.*?)</score>', result, re.DOTALL)
                score_text = score_match.group(1).strip() if score_match else "{}"
                try:
                    score_data = json.loads(score_text)
                except:
                    score_data = {}
                
                # Extract answer (from <answer> tag)
                answer_match = re.search(r'<answer>(.*?)</answer>', result, re.DOTALL)
                new_prompt = answer_match.group(1).strip() if answer_match else rewritten_prompt
                
                # Determine if satisfied (based on score or if prompt changed)
                # If new_prompt is the same as rewritten_prompt, usually indicates satisfied
                is_satisfied = (new_prompt.strip() == rewritten_prompt.strip()) or \
                              (score_data.get('semantic', 0) >= 8 and score_data.get('quality', 0) >= 8)
                
                # If no new prompt was generated, use original rewritten_prompt
                if not new_prompt or new_prompt.strip() == "":
                    new_prompt = rewritten_prompt
                    is_satisfied = True
                
                parsed_result = {
                    'is_satisfied': is_satisfied,
                    'reason': reason,
                    'new_rewritten_prompt': new_prompt if not is_satisfied else rewritten_prompt,
                    'score': score_data
                }
            else:
                # Parse default JSON format
                if isinstance(result, str):
                    result = result.replace('```json', '')
                    result = result.replace('```', '')
                    parsed_result = json.loads(result)
                else:
                    parsed_result = json.loads(result)
            
            # Validate required fields
            if 'is_satisfied' not in parsed_result or 'reason' not in parsed_result:
                raise ValueError("Response missing required fields")
            
            # Return result without printing (for cleaner logs)
            return parsed_result
            
        except Exception as e:
            print(f"[Warning] Error during evaluation API call (attempt {_ + 1}/{max_retries}): {e}")
            if _ == max_retries - 1:
                print(f"[Error] Failed to evaluate after {max_retries} attempts, marking as not satisfied with original rewritten_prompt")
                return {
                    'is_satisfied': False,
                    'reason': f'Evaluation failed after {max_retries} attempts',
                    'new_rewritten_prompt': rewritten_prompt if rewritten_prompt else original_prompt
                }



def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark Edit-Thinker inference script.")
    parser.add_argument(
        "--api_endpoint",
        type=str,
        required=True,
        help="API endpoint URL for image generation. Can be a single endpoint or comma-separated multiple endpoints for round-robin load balancing.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for authentication (if required).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name for output directory and API requests.",
    )
    parser.add_argument(
        "--rewrite_model_name",
        type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        help="Model name for rewrite (vLLM model name).",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=100,
        help="Maximum number of retries for API requests (default: 100)."
    )
    parser.add_argument(
        "--max_edit_turns",
        type=int,
        default=2,
        help="Maximum number of editing turns with refinement (default: 2)."
    )
    parser.add_argument(
        "--skip_first_turn_rewrite",
        action="store_true",
        help="Skip rewriting prompt in the first turn and use original instruction directly."
    )
    parser.add_argument(
        "--use_eval_logic_first_turn",
        action="store_true",
        help="Use evaluate_and_rewrite_prompt logic in the first turn with original image as edited image and empty rewritten prompt."
    )
    parser.add_argument(
        "--use_polish_edit_prompt_first_turn",
        action="store_true",
        help="Use polish_edit_prompt to rewrite the instruction in the first turn before editing."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="stepfun-ai/GEdit-Bench",
        help="Dataset name for loading data.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="data/image",
        help="Path to the image directory.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="data/result",
        help="Path to the result directory.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing (default: 1, sequential processing).",
    )
    parser.add_argument(
        "--polish_prompt_file",
        type=str,
        default=None,
        help="Path to the polish edit prompt file. If not specified, uses default prompt.",
    )
    parser.add_argument(
        "--eval_prompt_file",
        type=str,
        default=None,
        help="Path to the evaluation prompt file. If not specified, uses default prompt.",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=None,
        help="JSON file containing metadata for evaluation (optional). If specified, will load from metadata file instead of HuggingFace dataset.",
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        default=None,
        choices=["gedit", "imgedit", "kris", "rise"],
        help="Dataset format: 'gedit' (HuggingFace GEdit-Bench), 'imgedit', 'kris', or 'rise'. Required when using --metadata_file.",
    )
    return parser.parse_args()


def load_dataset_with_metadata(args):
    """Load dataset from either HuggingFace or local metadata file."""
    if args.metadata_file:
        # dataset_format must be explicitly specified when using metadata_file
        if args.dataset_format is None:
            raise ValueError("--dataset_format must be specified when using --metadata_file. Choose from: 'imgedit', 'kris', 'rise'")
        
        # Load from local metadata file
        with open(args.metadata_file, "r") as f:
            tmpdatas = json.load(f)
        
        metadatas = []
        
        if args.dataset_format == "kris":
            # KRIS format: list of dicts or dict with keys
            if isinstance(tmpdatas, list):
                for idx, item in enumerate(tmpdatas):
                    tmp = item.copy()
                    tmp['key'] = tmp.get('id', str(idx))
                    tmp['instruction'] = tmp.get('ins_en', '')
                    tmp['type'] = tmp.get('type', 'unknown')
                    tmp['dataset_format'] = 'kris'
                    metadatas.append(tmp)
            else:
                # Dict format
                for k, v in tmpdatas.items():
                    tmp = v.copy()
                    tmp['key'] = tmp.get('id', k)
                    tmp['instruction'] = tmp.get('ins_en', '')
                    tmp['type'] = tmp.get('type', 'unknown')
                    tmp['dataset_format'] = 'kris'
                    metadatas.append(tmp)
        
        elif args.dataset_format == "rise":
            # RISE format: list of dicts
            if isinstance(tmpdatas, list):
                for idx, item in enumerate(tmpdatas):
                    tmp = item.copy()
                    tmp['key'] = tmp.get('index', str(idx))
                    tmp['instruction'] = tmp.get('instruction', '')
                    tmp['category'] = tmp.get('category', 'unknown')
                    tmp['dataset_format'] = 'rise'
                    metadatas.append(tmp)
            else:
                # Dict format
                for k, v in tmpdatas.items():
                    tmp = v.copy()
                    tmp['key'] = tmp.get('index', k)
                    tmp['instruction'] = tmp.get('instruction', '')
                    tmp['category'] = tmp.get('category', 'unknown')
                    tmp['dataset_format'] = 'rise'
                    metadatas.append(tmp)
        
        else:
            # imgedit format (default): Bagel format
            for k, v in tmpdatas.items():
                tmp = v.copy()
                tmp['path'] = tmp.get('id', tmp.get('path', k))
                tmp['key'] = k
                tmp['instruction'] = tmp.get('prompt', tmp.get('instruction', ''))
                tmp['dataset_format'] = 'imgedit'
                metadatas.append(tmp)
        
        return metadatas
    else:
        # Load from HuggingFace dataset
        # If dataset_format not specified, default to gedit
        if args.dataset_format is None:
            args.dataset_format = "gedit"
        
        dataset = load_dataset(args.dataset_name)['train']
        # If GEdit format, filter by language
        if args.dataset_format == "gedit":
            # GEdit format: filter by language
            if hasattr(dataset, 'filter') and 'instruction_language' in dataset.column_names:
                dataset = dataset.filter(lambda x: x["instruction_language"] == "en", num_proc=4)
        return dataset


def process_metadata_format(metadata_item, image_path, dataset_format):
    """Process metadata and load image based on dataset format."""
    if dataset_format is None:
        raise ValueError("dataset_format must be specified")
    
    try:
        dataset_format = metadata_item.get('dataset_format', dataset_format)
        
        if dataset_format == "kris":
            # KRIS format: ori_img can be str or list, type is category
            ori_img = metadata_item.get('ori_img', '')
            img_type = metadata_item.get('type', 'unknown')
            
            if isinstance(ori_img, str):
                img_path = os.path.join(image_path, img_type, ori_img)
            elif isinstance(ori_img, list) and len(ori_img) > 0:
                # Use first image if multiple images
                img_path = os.path.join(image_path, img_type, ori_img[0])
            else:
                raise ValueError(f"Invalid ori_img format in KRIS metadata: {ori_img}")
            
            input_image = Image.open(img_path).convert('RGB')
            
            return {
                'key': metadata_item['key'],
                'instruction': metadata_item['instruction'],
                'input_image': ImageOps.exif_transpose(input_image),
                'type': img_type,
                'ori_img': ori_img,  # Save original ori_img field for subsequent path construction
                'dataset_format': 'kris',
            }
        
        elif dataset_format == "rise":
            # RISE format: image is filename, category is category
            image_filename = metadata_item.get('image', '')
            category = metadata_item.get('category', 'unknown')
            
            img_path = os.path.join(image_path, image_filename)
            input_image = Image.open(img_path).convert('RGB')
            
            return {
                'key': metadata_item['key'],
                'instruction': metadata_item['instruction'],
                'input_image': ImageOps.exif_transpose(input_image),
                'category': category,
                'image': image_filename,  # Save original image field for subsequent path construction
                'dataset_format': 'rise',
            }
        
        else:
            # imgedit format (default): Bagel format
            path = metadata_item.get('path', metadata_item.get('id', metadata_item['key']))
            img_path = os.path.join(image_path, path)
            input_image = Image.open(img_path).convert('RGB')
            
            return {
                'key': metadata_item['key'],
                'instruction': metadata_item['instruction'],
                'input_image': ImageOps.exif_transpose(input_image),
                'path': path,  # Save path field for subsequent path construction
                'dataset_format': 'imgedit',
            }
    
    except Exception as e:
        print(f"Error loading image for key {metadata_item.get('key', 'unknown')}: {e}")
        return None


def get_model_config(model_name: str) -> dict:
    if model_name == "flux-kontext-dev":
        return {
            'num_inference_step': 28,
            'guidance_scale': 2.5,
            'width': 1024,
            'height': 1024,
            'enhance_prompt': False,
            'seed': 0,
        }
    elif model_name == "qwen-image-edit":
        return {
            'num_inference_step': 50,
            'guidance_scale': 5.0,
            'width': 1024,
            'height': 1024,
            'enhance_prompt': False,
            # 'seed': 0,
        }
    elif model_name == "omnigen2-edit":
        return {
            'num_inference_step': 50,
            'text_guidance_scale': 5.0,
            'image_guidance_scale': 1.5,
            'width': 1024,
            'height': 1024,
            'negative_prompt': "",
            'enhance_prompt': False,
            'seed': 0,
        }
    elif model_name == "longcat-image-edit":
        return {
            'num_inference_step': 50,
            'guidance_scale': 4.5,
            'width': 1024,
            'height': 1024,
            'enhance_prompt': False,
            'negative_prompt': "",
            'seed': 0,
        }
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def image_to_base64(image_path, max_size=None):
    if max_size is not None:
        """Convert image file to base64 string with max size limit"""
        with Image.open(image_path) as img:
            width, height = img.size
            max_side = max(width, height)
            if max_side > max_size:
                # Calculate scaling ratio
                scale = max_size / max_side
                new_size = (int(width * scale), int(height * scale))
                img = img.resize(new_size, Image.LANCZOS)
            # Save to memory
            buffer = io.BytesIO()
            img.save(buffer, format=img.format if img.format else "PNG")
            base64_str = base64.b64encode(buffer.getvalue()).decode()
            return base64_str
    else:
        """Convert image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode() 


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

def check_api_health(api_endpoint: str) -> bool:
    """Check API server health status"""
    try:
        # Try to access health check endpoint
        health_url = api_endpoint + "/health"
        response = requests.get(health_url, timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to check API server health: {str(e)}")
        return False

def call_api_with_retry(endpoint_pool: EndpointPool,
                        instruction: str, 
                        input_images: List[Image.Image],
                        model_config: dict,
                        max_retries: int = 100) -> Image.Image:
    """Call API for image editing/generation with retry logic and endpoint polling"""
    
    # Encode input image (usually editing API only needs one image)
    image_b64 = None
    if input_images and len(input_images) > 0:
        image_b64 = encode_image_to_base64(input_images[0])
    
    # Prepare request data (following FLUX API format)
    request_data = {
        "image": image_b64,  # base64 encoded input image
        "prompt": instruction,
        **model_config,
    }
    
    # Retry logic
    for attempt in range(max_retries):
        try:
            # Get next endpoint from pool (round-robin)
            current_endpoint = endpoint_pool.get_next_endpoint()
            
            response = requests.post(
                f"{current_endpoint}/edit",
                json=request_data,
                timeout=600  # 10 minute timeout (image editing may take longer)
            )
            result = response.json()

            if result["success"]:
                output_image = decode_base64_to_image(result["edited_image"])
                return output_image
            else:
                raise Exception(result["error"])
                    
        except Exception as e:
            print(f"API request error (endpoint: {current_endpoint}, attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                raise Exception(f"Failed to generate image after {max_retries} attempts")


def run(endpoint_pool: EndpointPool,
        instruction: str, 
        input_images: List[Image.Image],
        model_config: dict,
        max_retries: int = 100):
    """Call API for image generation"""
    return call_api_with_retry(endpoint_pool, instruction, input_images, model_config, max_retries)


def process_single_sample(data_item, args, model_config, openai_client_config, endpoint_pool_config, polish_prompt_template=None, eval_prompt_template=None):
    """
    Function to process a single sample, for parallel execution
    
    Args:
        data_item: Single sample from dataset
        args: Command line arguments
        model_config: Model configuration
        openai_client_config: OpenAI client configuration (base_url, api_key)
        endpoint_pool_config: Endpoint pool configuration (endpoints list)
    
    Returns:
        dict: Dictionary containing processing status and result path
    """
    # Recreate OpenAI client in subprocess
    openai_client = OpenAIClient(
        api_key=openai_client_config['api_key'],
        base_url=openai_client_config['base_url']
    )
    
    # Recreate endpoint pool in subprocess
    endpoint_pool = EndpointPool(endpoints=endpoint_pool_config['endpoints'])
    
    # Support multiple formats: GEdit format, imgedit format, KRIS format, RISE format
    key = data_item['key']
    original_instruction = data_item['instruction']
    
    # Get dataset format (from data_item or args)
    dataset_format = data_item.get('dataset_format', args.dataset_format)
    
    if dataset_format is None:
        raise ValueError("dataset_format must be specified. Use --dataset_format argument.")
    
    if dataset_format == "gedit":
        # GEdit format: Load from HuggingFace dataset
        task_type = data_item['task_type']
        instruction_language = data_item['instruction_language']
        input_image = data_item['input_image']  # PIL Image
        input_image_path = os.path.join(args.image_path, f"{key}.png")
    elif dataset_format == "kris":
        # KRIS format
        task_type = data_item.get('type', 'unknown')
        instruction_language = "en"  # KRIS defaults to English
        input_image = data_item['input_image']  # PIL Image (already loaded via process_metadata_format)
        # Build image path: if ori_img is list, take first element; if string, use directly
        ori_img = data_item.get('ori_img', f"{key}.png")
        if isinstance(ori_img, list) and len(ori_img) > 0:
            ori_img = ori_img[0]
        input_image_path = os.path.join(args.image_path, task_type, ori_img)
    elif dataset_format == "rise":
        # RISE format
        task_type = data_item.get('category', 'unknown')
        instruction_language = "en"  # RISE defaults to English
        input_image = data_item['input_image']  # PIL Image (already loaded via process_metadata_format)
        input_image_path = os.path.join(args.image_path, data_item.get('image', f"{key}.png"))
    else:
        # imgedit format: Load from metadata file, input_image may be PIL Image or needs to be loaded from path
        task_type = "imgedit"  # default value
        instruction_language = "en"  # default value
        if 'input_image' in data_item and isinstance(data_item['input_image'], Image.Image):
            input_image = data_item['input_image']
        elif 'path' in data_item:
            # Load image from path
            input_image = Image.open(os.path.join(args.image_path, data_item['path'])).convert('RGB')
            input_image = ImageOps.exif_transpose(input_image)
        else:
            # Try to use key as filename
            input_image = Image.open(os.path.join(args.image_path, f"{key}.png")).convert('RGB')
            input_image = ImageOps.exif_transpose(input_image)
        # Build correct image path: use path field (if exists), otherwise use key
        if 'path' in data_item:
            input_image_path = os.path.join(args.image_path, data_item['path'])
        else:
            input_image_path = os.path.join(args.image_path, f"{key}.png")
    
    # Process input image
    ori_img_size = input_image.size
    new_img_size = (ori_img_size[0] // 16 * 16, ori_img_size[1] // 16 * 16)
    resized_input_image = input_image.resize(new_img_size)
    input_images = [resized_input_image]

    # Create final output directory
    if dataset_format == "gedit":
        final_output_dir = os.path.join(args.result_dir, "final", "fullset", task_type, instruction_language)
    elif dataset_format == "kris":
        # KRIS format: categorize by type
        final_output_dir = os.path.join(args.result_dir, "final", task_type)
    elif dataset_format == "rise":
        # RISE format: categorize by category
        final_output_dir = os.path.join(args.result_dir, "final", task_type)
    else:
        # imgedit format: use simple directory structure
        final_output_dir = os.path.join(args.result_dir, "final", 'gen_image')
    os.makedirs(final_output_dir, exist_ok=True)
    
    final_output_path = os.path.join(final_output_dir, f"{key}.png")
    final_source_path = os.path.join(final_output_dir, f"{key}_SRCIMG.png")

    # Skip if already exists
    if os.path.exists(final_output_path):
        return {
            'status': 'skipped',
            'key': key,
            'reason': 'already_exists'
        }

    # Multi-turn editing and refinement
    current_prompt = None
    output_image = None
    is_satisfied = False
    
    # Initialize current sample metadata
    meta_data = {
        "key": key,
        "task_type": task_type,
        "instruction_language": instruction_language,
        "original_instruction": original_instruction,
        "max_turns": args.max_edit_turns,
        "turns": [],
        "is_satisfied": False,
        "final_turn": None
    }
            
    eval_rewrite_func = evaluate_and_rewrite_prompt
    
    for turn in range(args.max_edit_turns):
        turn_num = turn + 1
        
        
        # Create directory for current turn
        if dataset_format == "gedit":
            turn_dir = os.path.join(args.result_dir, f"turn{turn_num}", "fullset", task_type, instruction_language)
        elif dataset_format == "kris":
            # KRIS format: categorize by type
            turn_dir = os.path.join(args.result_dir, f"turn{turn_num}", task_type)
        elif dataset_format == "rise":
            # RISE format: categorize by category
            turn_dir = os.path.join(args.result_dir, f"turn{turn_num}", task_type)
        else:
            # imgedit format: use simple directory structure
            turn_dir = os.path.join(args.result_dir, f"turn{turn_num}", 'gen_image')
        os.makedirs(turn_dir, exist_ok=True)
        
        turn_output_path = os.path.join(turn_dir, f"{key}.png")
        turn_source_path = os.path.join(turn_dir, f"{key}_SRCIMG.png")

        if is_satisfied:
            # Only save image to current turn (if we already have output_image from previous turn)
            if output_image is not None:
                if output_image.size != ori_img_size:
                    output_image.resize(ori_img_size).save(turn_output_path)
                else:
                    output_image.save(turn_output_path)
            continue
        
        # Save source image for each turn
        input_image.save(turn_source_path)
        
        # Initialize turn metadata
        turn_meta = {
            "turn_number": turn_num,
            "rewritten_prompt": None,
            "evaluation": None,
            "generated": False,
            "output_path": turn_output_path,
            "source_path": turn_source_path
        }
        
        if turn == 0:
            # First turn: use polish_edit_prompt to rewrite or use original instruction
            if args.use_eval_logic_first_turn:
                # Use evaluate_and_rewrite_prompt logic, with original image as edited image
                eval_result = eval_rewrite_func(
                    original_image_path=input_image_path,
                    original_prompt=original_instruction,
                    rewritten_prompt="",  # Empty rewritten prompt for first turn
                    edited_image_path=input_image_path,  # Use original image as edited image
                    openai_client=openai_client,
                    model=args.rewrite_model_name,
                    prompt_template=eval_prompt_template
                )
                
                turn_meta["method"] = f"evaluate_and_rewrite_prompt_first_turn"
                turn_meta["evaluation"] = {
                    "is_satisfied": eval_result['is_satisfied'],
                    "reason": eval_result['reason'],
                    "new_prompt": eval_result.get('new_rewritten_prompt')
                }
                
                if eval_result['is_satisfied']:
                    is_satisfied = True
                    # Use original image as output
                    input_image.save(turn_output_path)
                    output_image = input_image
                    turn_meta["generated"] = False
                    turn_meta["action"] = "original_image_satisfied"
                    turn_meta["rewritten_prompt"] = ""
                    meta_data["turns"].append(turn_meta)
                    break
                else:
                    current_prompt = eval_result['new_rewritten_prompt']
                    turn_meta["rewritten_prompt"] = current_prompt
                    if current_prompt is None:
                        # Use original image as output
                        input_image.save(turn_output_path)
                        output_image = input_image
                        turn_meta["generated"] = False
                        turn_meta["action"] = "no_refined_prompt_stopping"
                        meta_data["turns"].append(turn_meta)
                        break
            elif args.use_polish_edit_prompt_first_turn:
                # Use polish_edit_prompt to rewrite instruction
                current_prompt = polish_edit_prompt(
                    prompt=original_instruction,
                    img=input_image_path,
                    openai_client=openai_client,
                    model=args.rewrite_model_name,
                    prompt_template=polish_prompt_template
                )
                turn_meta["rewritten_prompt"] = current_prompt
                turn_meta["method"] = "polish_edit_prompt"
                turn_meta["original_prompt"] = original_instruction
            elif args.skip_first_turn_rewrite:
                current_prompt = original_instruction
                turn_meta["rewritten_prompt"] = current_prompt
                turn_meta["method"] = "no_rewrite"
            else:
                # If polish_edit_prompt not defined, use original instruction
                current_prompt = original_instruction
                turn_meta["rewritten_prompt"] = current_prompt
                turn_meta["method"] = "no_rewrite"
        else:
            # Subsequent turns: use evaluate_and_rewrite_prompt
            # Get output from previous turn for evaluation
            if dataset_format == "gedit":
                prev_turn_dir = os.path.join(args.result_dir, f"turn{turn}", "fullset", task_type, instruction_language)
            elif dataset_format == "kris":
                prev_turn_dir = os.path.join(args.result_dir, f"turn{turn}", task_type)
            elif dataset_format == "rise":
                prev_turn_dir = os.path.join(args.result_dir, f"turn{turn}", task_type)
            else:
                prev_turn_dir = os.path.join(args.result_dir, f"turn{turn}", 'gen_image')
            prev_output_path = os.path.join(prev_turn_dir, f"{key}.png")
            
            eval_result = eval_rewrite_func(
                original_image_path=input_image_path,
                original_prompt=original_instruction,
                rewritten_prompt=current_prompt,
                edited_image_path=prev_output_path,
                openai_client=openai_client,
                model=args.rewrite_model_name,
                prompt_template=eval_prompt_template
            )
            
            turn_meta["method"] = f"evaluate_and_rewrite_prompt"
            turn_meta["evaluation"] = {
                "is_satisfied": eval_result['is_satisfied'],
                "reason": eval_result['reason'],
                "new_prompt": eval_result.get('new_rewritten_prompt')
            }
            
            if eval_result['is_satisfied']:
                is_satisfied = True
                # Copy previous turn result as current turn output
                prev_image = Image.open(prev_output_path)
                prev_image.save(turn_output_path)
                output_image = prev_image
                turn_meta["generated"] = False
                turn_meta["action"] = "copied_from_previous_turn"
                meta_data["turns"].append(turn_meta)
                break  # Stop processing, satisfied with current result
            else:
                current_prompt = eval_result['new_rewritten_prompt']
                turn_meta["rewritten_prompt"] = current_prompt
                if current_prompt is None:
                    # Copy previous turn result as current turn output
                    prev_image = Image.open(prev_output_path)
                    prev_image.save(turn_output_path)
                    output_image = prev_image
                    turn_meta["generated"] = False
                    turn_meta["action"] = "no_refined_prompt_stopping"
                    meta_data["turns"].append(turn_meta)
                    break  # Stop processing, no refined prompt available
        
        # Generate image using current prompt via API (only if not satisfied and have valid prompt)
        if current_prompt:
            output_image = run(endpoint_pool, current_prompt, input_images, model_config, args.max_retries)
            
            # Save result of current turn
            output_image_resized = output_image.resize(ori_img_size)
            output_image_resized.save(turn_output_path)
            turn_meta["generated"] = True
            turn_meta["action"] = "generated"
            
            # Add turn metadata
            meta_data["turns"].append(turn_meta)
    
    # Save final result (best result from all turns)
    # Ensure output_image is not None (if ended early, output_image should be set before break)
    if output_image is None:
        # If output_image not set, use input image as final output (shouldn't happen, but as safety measure)
        output_image = input_image
        print(f"[WARNING] output_image is None for key {key}, using input_image as fallback")
    
    if output_image.size != ori_img_size:
        final_output = output_image.resize(ori_img_size)
    else:
        final_output = output_image
    final_output.save(final_output_path)
    
    # Save source image to final directory
    input_image.save(final_source_path)
    
    # Copy all turn results to final directory with turn suffix
    # Only copy actually executed turns (recorded in meta_data["turns"])
    turn_files_in_final = []
    for turn_info in meta_data["turns"]:
        turn_num = turn_info["turn_number"]
        turn_output_src = turn_info["output_path"]
        
        if os.path.exists(turn_output_src):
            # Copy to final directory, adding _turn{i} suffix
            turn_final_path = os.path.join(final_output_dir, f"{key}_turn{turn_num}.png")
            turn_image = Image.open(turn_output_src)
            turn_image.save(turn_final_path)
            turn_files_in_final.append({
                "turn": turn_num,
                "path": turn_final_path
            })
        else:
            # If turn output file doesn't exist, log warning
            print(f"[WARNING] Turn {turn_num} output file not found: {turn_output_src}")
    
    # Update final metadata
    meta_data["is_satisfied"] = is_satisfied
    meta_data["final_turn"] = len(meta_data["turns"])
    meta_data["final_output_path"] = final_output_path
    meta_data["final_source_path"] = final_source_path
    meta_data["all_turns_in_final"] = turn_files_in_final
    
    # Save metadata to JSON file
    meta_json_path = os.path.join(final_output_dir, f"{key}_meta.json")
    with open(meta_json_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)
    
    return {
        'status': 'success',
        'key': key,
        'is_satisfied': is_satisfied,
        'final_turn': len(meta_data["turns"]),
        'output_path': final_output_path
    }

def main(args: argparse.Namespace, openai_client) -> None:
    """Main function to run image generation pipeline"""
    
    print("=" * 70)
    print("🎨 Image Editing API Call Script")
    print("=" * 70)
    
    # Parse API endpoints (supports comma-separated multiple endpoints)
    endpoint_list = [ep.strip() for ep in args.api_endpoint.split(',')]
    
    # Print endpoint info
    print(f"📡 Initializing API endpoint pool with {len(endpoint_list)} endpoints:")
    for i, endpoint in enumerate(endpoint_list):
        print(f"  [{i+1}] {endpoint}")
    
    # Print API configuration
    print(f"\n📡 API Configuration:")
    if len(endpoint_list) > 1:
        print(f"  Endpoint pool mode: Round-robin {len(endpoint_list)} endpoints")
    else:
        print(f"  Endpoint: {endpoint_list[0]}")
    print(f"  Model: {args.model_name}")
    print(f"  Max retries: {args.max_retries}")
    print(f"  Parallel workers: {args.num_workers}")

    model_config = get_model_config(args.model_name)
    print(f"  Model config: {model_config}")
    
    # Check API server health
    print(f"\n🔍 Checking API server status...")
    healthy_count = 0
    for endpoint in endpoint_list:
        if check_api_health(endpoint):
            print(f"  ✅ {endpoint} responding normally")
            healthy_count += 1
        else:
            print(f"  ⚠️  {endpoint} not accessible")
    
    if healthy_count > 0:
        print(f"\n  Total: {healthy_count}/{len(endpoint_list)} endpoints available")
    else:
        print(f"\n  ⚠️  All endpoints not accessible (will continue attempting calls)")
    
    # Load dataset
    print(f"\n📦 Loading dataset")
    if args.metadata_file:
        print(f"  Loading from metadata file: {args.metadata_file}")
        raw_dataset = load_dataset_with_metadata(args)
        # Process metadata format and load images
        test_dataset = []
        for item in raw_dataset:
            processed_item = process_metadata_format(item, args.image_path, args.dataset_format)
            if processed_item is not None:
                test_dataset.append(processed_item)
        print(f"  Dataset size: {len(test_dataset)} samples")
        print(f"  Dataset format: {args.dataset_format}")
    else:
        print(f"  Loading from HuggingFace: {args.dataset_name}")
        test_dataset = load_dataset_with_metadata(args)
        print(f"  Dataset size: {len(test_dataset)} samples")
        print(f"  Dataset format: {args.dataset_format}")
    print(f"  Output directory: {args.result_dir}")
    print(f"  Max edit turns: {args.max_edit_turns}")
    
    # Print first turn rewrite strategy
    print(f"\n📝 First turn rewrite strategy:")
    if args.use_eval_logic_first_turn:
        print(f"  ✓ Use evaluate_and_rewrite_prompt to evaluate original image")
    elif args.use_polish_edit_prompt_first_turn:
        print(f"  ✓ Use polish_edit_prompt to rewrite instruction")
    elif args.skip_first_turn_rewrite:
        print(f"  ✓ Skip rewrite, use original instruction")
    else:
        print(f"  ✓ Default mode (use original instruction)")
    
    print("-" * 70)
    
    # Load prompt files
    polish_prompt_template = None
    eval_prompt_template = None
    
    if args.polish_prompt_file:
        polish_prompt_template = load_prompt_from_file(args.polish_prompt_file)
        if polish_prompt_template:
            print(f"\n📄 Loaded polish prompt file: {args.polish_prompt_file}")
        else:
            print(f"\n⚠️  Failed to load polish prompt file: {args.polish_prompt_file}, will use default prompt")
    
    if args.eval_prompt_file:
        eval_prompt_template = load_prompt_from_file(args.eval_prompt_file)
        if eval_prompt_template:
            print(f"📄 Loaded eval prompt file: {args.eval_prompt_file}")
        else:
            print(f"⚠️  Failed to load eval prompt file: {args.eval_prompt_file}, will use default prompt")
    
    # Prepare OpenAI client configuration (for passing to subprocesses)
    openai_client_config = {
        'base_url': openai_client.base_url,
        'api_key': openai_client.api_key,
    }
    
    # Prepare endpoint pool configuration (for passing to subprocesses)
    endpoint_pool_config = {
        'endpoints': endpoint_list,
    }
    
    # Decide whether to use sequential or parallel processing based on num_workers
    if args.num_workers <= 1:
        # Sequential processing
        print(f"\n🔄 Using sequential processing mode")
        results = []
        for data in tqdm(test_dataset, desc="Generating images", unit="sample"):
            result = process_single_sample(
                data, args, model_config, openai_client_config, endpoint_pool_config,
                polish_prompt_template=polish_prompt_template,
                eval_prompt_template=eval_prompt_template
            )
            results.append(result)
    else:
        # Parallel processing
        print(f"\n🔄 Using parallel processing mode (workers: {args.num_workers})")
        results = []
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Create partial function with fixed args, model_config, openai_client_config, endpoint_pool_config
            process_func = partial(
                process_single_sample,
                args=args,
                model_config=model_config,
                openai_client_config=openai_client_config,
                endpoint_pool_config=endpoint_pool_config,
                polish_prompt_template=polish_prompt_template,
                eval_prompt_template=eval_prompt_template
            )
            
            # Submit all tasks
            futures = {executor.submit(process_func, data): data for data in test_dataset}
            
            # Show progress with tqdm
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating images", unit="sample"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    data = futures[future]
                    print(f"\n⚠️ Error processing sample {data.get('key', 'unknown')}: {str(e)}")
                    results.append({
                        'status': 'error',
                        'key': data.get('key', 'unknown'),
                        'error': str(e)
                    })
    
    # Statistics and summary
    print("\n" + "=" * 70)
    print("📊 Processing Results Summary:")
    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    print(f"  ✅ Success: {success_count} samples")
    print(f"  ⏭️  Skipped: {skipped_count} samples")
    print(f"  ❌ Error: {error_count} samples")
    print(f"  📝 Total: {len(results)} samples")
    
    if success_count > 0:
        satisfied_count = sum(1 for r in results if r['status'] == 'success' and r.get('is_satisfied', False))
        print(f"  ✨ Satisfied: {satisfied_count}/{success_count} samples")
    
    print("=" * 70)
    
    # Save summary statistics
    summary_path = os.path.join(args.result_dir, "final", "processing_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total': len(results),
            'success': success_count,
            'skipped': skipped_count,
            'error': error_count,
            'satisfied': sum(1 for r in results if r['status'] == 'success' and r.get('is_satisfied', False)),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Processing summary saved to: {summary_path}\n")

if __name__ == "__main__":
    base_url = os.getenv('VLLM_BASE_URL', None)
    api_key = os.getenv('VLLM_API_KEY', 'EMPTY')

    if base_url is None:
        print("Using OpenAI API")
        # Please replace with your own API endpoint
        base_url = "https://your-endpoint.com/v1/openai/native"  # Replace with your API endpoint
        api_key = os.getenv('OPENAI_API_KEY')
    else:
        print(f"Using vLLM API:")
        print(f"  Base URL: {base_url}")
        print(f"  API Key: {'Set' if api_key != 'EMPTY' else 'EMPTY (default)'}")
        print("***********************")
    openai_client = OpenAIClient(
        api_key=api_key,
        base_url=base_url
    )
    args = parse_args()
    main(args, openai_client)