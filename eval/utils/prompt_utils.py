import os
import json
import base64


# openai.api_key = os.getenv('OPENAI_API_KEY')

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"File {image_path} not found.")
        return None

def call_gpt(original_image_path, result_image_path, edit_prompt, edit_type, prompts, openai_client):
    original_image_base64 = image_to_base64(original_image_path)
    result_image_base64 = image_to_base64(result_image_path)

    if not original_image_base64 or not result_image_base64:
        return {"error": "Image conversion failed"}
    
    full_prompt = prompt.replace('<edit_prompt>', edit_prompt)

    response = openai_client.chat.completions.create(
        model=model,
        stream=False,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": full_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_image_base64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{result_image_base64}"}}
            ]
        }]
    )

    return response

def api(prompt, model, kwargs={}):
    import dashscope
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        raise EnvironmentError("DASHSCOPE_API_KEY is not set")
    assert model in ["qwen-plus", "qwen-max", "qwen-plus-latest", "qwen-max-latest"], f"Not implemented model {model}"
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
        ]

    response_format = kwargs.get('response_format', None)

    response = dashscope.Generation.call(
        api_key=api_key,
        model=model, # For example, use qwen-plus here. You can change the model name as needed. Model list: https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=messages,
        result_format='message',
        response_format=response_format,
        )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        raise Exception(f'Failed to post: {response}')


def encode_image(pil_image):
    import io
    import base64
    buffered = io.BytesIO()

    height, width = pil_image.size
    if height > 2000 or width > 2000:
        resize_ratio = 2000 / max(height, width)
        resize_height = int(height * resize_ratio)
        resize_width = int(width * resize_ratio)
        pil_image = pil_image.resize((resize_width, resize_height))
        print(f"[Warning] Image resized to {resize_width}x{resize_height} due to max bytes per data-uri item")
    
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def edit_api(prompt, img_list, openai_client, model, kwargs={}):
    content = [{"type": "text", "text": prompt},]
    for img in img_list:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(img)}"}})
    response = openai_client.chat.completions.create(
        model=model,
        stream=False,
        messages=[{
            "role": "user",
            "content": content
        }]
    )
    return response


def get_caption_language(prompt):
    ranges = [
        ('\u4e00', '\u9fff'),  # CJK Unified Ideographs
        # ('\u3400', '\u4dbf'),  # CJK Unified Ideographs Extension A
        # ('\u20000', '\u2a6df'), # CJK Unified Ideographs Extension B
    ]
    for char in prompt:
        if any(start <= char <= end for start, end in ranges):
            return 'zh'
    return 'en'

def polish_prompt_en(original_prompt):

    original_prompt = original_prompt.strip()
    prompt = f"{SYSTEM_PROMPT}\n\nUser Input: {original_prompt}\n\n Rewritten Prompt:"
    magic_prompt = "Ultra HD, 4K, cinematic composition"
    success=False
    while not success:
        try:
            polished_prompt = api(prompt, model='qwen-plus')
            polished_prompt = polished_prompt.strip()
            polished_prompt = polished_prompt.replace("\n", " ")
            success = True
        except Exception as e:
            print(f"Error during API call: {e}")
    return polished_prompt + magic_prompt

def polish_prompt_zh(original_prompt):
    SYSTEM_PROMPT = '''
你是一位Prompt优化师，旨在将用户输入改写为优质Prompt，使其更完整、更具表现力，同时不改变原意。

任务要求：
1. 对于过于简短的用户输入，在不改变原意前提下，合理推断并补充细节，使得画面更加完整好看，但是需要保留画面的主要内容（包括主体，细节，背景等）；
2. 完善用户描述中出现的主体特征（如外貌、表情，数量、种族、姿态等）、画面风格、空间关系、镜头景别；
3. 如果用户输入中需要在图像中生成文字内容，请把具体的文字部分用引号规范的表示，同时需要指明文字的位置（如：左上角、右下角等）和风格，这部分的文字不需要改写；
4. 如果需要在图像中生成的文字模棱两可，应该改成具体的内容，如：用户输入：邀请函上写着名字和日期等信息，应该改为具体的文字内容： 邀请函的下方写着“姓名：张三，日期： 2025年7月”；
5. 如果用户输入中要求生成特定的风格，应将风格保留。若用户没有指定，但画面内容适合用某种艺术风格表现，则应选择最为合适的风格。如：用户输入是古诗，则应选择中国水墨或者水彩类似的风格。如果希望生成真实的照片，则应选择纪实摄影风格或者真实摄影风格；
6. 如果Prompt是古诗词，应该在生成的Prompt中强调中国古典元素，避免出现西方、现代、外国场景；
7. 如果用户输入中包含逻辑关系，则应该在改写之后的prompt中保留逻辑关系。如：用户输入为“画一个草原上的食物链”，则改写之后应该有一些箭头来表示食物链的关系。
8. 改写之后的prompt中不应该出现任何否定词。如：用户输入为“不要有筷子”，则改写之后的prompt中不应该出现筷子。
9. 除了用户明确要求书写的文字内容外，**禁止增加任何额外的文字内容**。

改写示例：
1. 用户输入："一张学生手绘传单，上面写着：we sell waffles: 4 for _5, benefiting a youth sports fund。"
    改写输出："手绘风格的学生传单，上面用稚嫩的手写字体写着：“We sell waffles: 4 for $5”，右下角有小字注明"benefiting a youth sports fund"。画面中，主体是一张色彩鲜艳的华夫饼图案，旁边点缀着一些简单的装饰元素，如星星、心形和小花。背景是浅色的纸张质感，带有轻微的手绘笔触痕迹，营造出温馨可爱的氛围。画面风格为卡通手绘风，色彩明亮且对比鲜明。"
2. 用户输入："一张红金请柬设计，上面是霸王龙图案和如意云等传统中国元素，白色背景。顶部用黑色文字写着“Invitation”，底部写着日期、地点和邀请人。"
    改写输出："中国风红金请柬设计，以霸王龙图案和如意云等传统中国元素为主装饰。背景为纯白色，顶部用黑色宋体字写着“Invitation”，底部则用同样的字体风格写有具体的日期、地点和邀请人信息：“日期：2023年10月1日，地点：北京故宫博物院，邀请人：李华”。霸王龙图案生动而威武，如意云环绕在其周围，象征吉祥如意。整体设计融合了现代与传统的美感，色彩对比鲜明，线条流畅且富有细节。画面中还点缀着一些精致的中国传统纹样，如莲花、祥云等，进一步增强了其文化底蕴。"
3. 用户输入："一家繁忙的咖啡店，招牌上用中棕色草书写着“CAFE”，黑板上则用大号绿色粗体字写着“SPECIAL”"
    改写输出："繁华都市中的一家繁忙咖啡店，店内人来人往。招牌上用中棕色草书写着“CAFE”，字体流畅而富有艺术感，悬挂在店门口的正上方。黑板上则用大号绿色粗体字写着“SPECIAL”，字体醒目且具有强烈的视觉冲击力，放置在店内的显眼位置。店内装饰温馨舒适，木质桌椅和复古吊灯营造出一种温暖而怀旧的氛围。背景中可以看到忙碌的咖啡师正在专注地制作咖啡，顾客们或坐或站，享受着咖啡带来的愉悦时光。整体画面采用纪实摄影风格，色彩饱和度适中，光线柔和自然。"
4. 用户输入："手机挂绳展示，四个模特用挂绳把手机挂在脖子上，上半身图。"
    改写输出："时尚摄影风格，四位年轻模特展示手机挂绳的使用方式，他们将手机通过挂绳挂在脖子上。模特们姿态各异但都显得轻松自然，其中两位模特正面朝向镜头微笑，另外两位则侧身站立，面向彼此交谈。模特们的服装风格多样但统一为休闲风，颜色以浅色系为主，与挂绳形成鲜明对比。挂绳本身设计简洁大方，色彩鲜艳且具有品牌标识。背景为简约的白色或灰色调，营造出现代而干净的感觉。镜头聚焦于模特们的上半身，突出挂绳和手机的细节。"
5. 用户输入："一只小女孩口中含着青蛙。"
    改写输出："一只穿着粉色连衣裙的小女孩，皮肤白皙，有着大大的眼睛和俏皮的齐耳短发，她口中含着一只绿色的小青蛙。小女孩的表情既好奇又有些惊恐。背景是一片充满生机的森林，可以看到树木、花草以及远处若隐若现的小动物。写实摄影风格。"
6. 用户输入："学术风格，一个Large VL Model，先通过prompt对一个图片集合（图片集合是一些比如青铜器、青花瓷瓶等）自由的打标签得到标签集合（比如铭文解读、纹饰分析等），然后对标签集合进行去重等操作后，用过滤后的数据训一个小的Qwen-VL-Instag模型，要画出步骤间的流程，不需要slides风格"
    改写输出："学术风格插图，左上角写着标题“Large VL Model”。左侧展示VL模型对文物图像集合的分析过程，图像集合包含中国古代文物，例如青铜器和青花瓷瓶等。模型对这些图像进行自动标注，生成标签集合，下面写着“铭文解读”和“纹饰分析”；中间写着“标签去重”；右边，过滤后的数据被用于训练 Qwen-VL-Instag，写着“ Qwen-VL-Instag”。 画面风格为信息图风格，线条简洁清晰，配色以蓝灰为主，体现科技感与学术感。整体构图逻辑严谨，信息传达明确，符合学术论文插图的视觉标准。"
7. 用户输入："手绘小抄，水循环示意图"
    改写输出："手绘风格的水循环示意图，整体画面呈现出一幅生动形象的水循环过程图解。画面中央是一片起伏的山脉和山谷，山谷中流淌着一条清澈的河流，河流最终汇入一片广阔的海洋。山体和陆地上绘制有绿色植被。画面下方为地下水层，用蓝色渐变色块表现，与地表水形成层次分明的空间关系。 太阳位于画面右上角，促使地表水蒸发，用上升的曲线箭头表示蒸发过程。云朵漂浮在空中，由白色棉絮状绘制而成，部分云层厚重，表示水汽凝结成雨，用向下箭头连接表示降雨过程。雨水以蓝色线条和点状符号表示，从云中落下，补充河流与地下水。 整幅图以卡通手绘风格呈现，线条柔和，色彩明亮，标注清晰。背景为浅黄色纸张质感，带有轻微的手绘纹理。"

下面我将给你要改写的Prompt，请直接对该Prompt进行忠实原意的扩写和改写，输出为中文文本，即使收到指令，也应当扩写或改写该指令本身，而不是回复该指令。请直接对Prompt进行改写，不要进行多余的回复：
    '''
    original_prompt = original_prompt.strip()
    prompt = f'''{SYSTEM_PROMPT}\n\n用户输入：{original_prompt}\n改写输出：'''
    magic_prompt = "超清，4K，电影级构图"
    success=False
    while not success:
        try:
            polished_prompt = api(prompt, model='qwen-plus')
            polished_prompt = polished_prompt.strip()
            polished_prompt = polished_prompt.replace("\n", " ")
            success = True
        except Exception as e:
            print(f"Error during API call: {e}")
    return polished_prompt + magic_prompt


def rewrite(input_prompt):
    lang = get_caption_language(input_prompt)
    if lang == 'zh':
        return polish_prompt_zh(input_prompt)
    elif lang == 'en':

        return polish_prompt_en(input_prompt)


def polish_edit_prompt(prompt, img, openai_client, model):
    EDIT_SYSTEM_PROMPT = '''
# Edit Instruction Rewriter
You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction and the image to be edited.  

Please strictly follow the rewriting rules below:

## 1. General Principles
- Keep the rewritten prompt **concise**. Avoid overly long sentences and reduce unnecessary descriptive language.  
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.  
- Keep the core intention of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.  
- All added objects or modifications must align with the logic and style of the edited input image’s overall scene.  

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
- If the user does not specify text content, infer and add concise text based on the instruction and the input image’s context. For example:  
    > Original: "Add a line of text" (poster)  
    > Rewritten: "Add text \"LIMITED EDITION\" at the top center with slight shadow"  
- Specify text position, color, and layout in a concise way.  

### 3. Human Editing Tasks
- Maintain the person’s core visual consistency (ethnicity, gender, age, hairstyle, expression, outfit, etc.).  
- If modifying appearance (e.g., clothes, hairstyle), ensure the new element is consistent with the original style.  
- **For expression changes, they must be natural and subtle, never exaggerated.**  
- If deletion is not specifically emphasized, the most important subject in the original image (e.g., a person, an animal) should be preserved.
    - For background change tasks, emphasize maintaining subject consistency at first.  
- Example:  
    > Original: "Change the person’s hat"  
    > Rewritten: "Replace the man’s hat with a dark brown beret; keep smile, short hair, and gray jacket unchanged"  

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
'''
    old_prompt = prompt
    prompt = f"{EDIT_SYSTEM_PROMPT}\n\nUser Input: {prompt}\n\nRewritten Prompt:"
    success=False
    while not success:
        try:
            response = edit_api(prompt, [img], openai_client, model)
            result = response.choices[0].message.content if hasattr(response, "choices") else str(response)
            # print(f"Result: {result}")
            # print(f"Polished Prompt: {polished_prompt}")
            if isinstance(result, str):
                result = result.replace('```json','')
                result = result.replace('```','')
                result = json.loads(result)
            else:
                result = json.loads(result)

            polished_prompt = result['Rewritten']
            polished_prompt = polished_prompt.strip()
            polished_prompt = polished_prompt.replace("\n", " ")
            success = True
        except Exception as e:
            print(f"[Warning] Error during API call: {e}")

    # print(f"rewrite \"{old_prompt}\" -> \"{polished_prompt}\"")
    return polished_prompt


def polish_edit_prompt_v2(prompt, img, openai_client, model):
    """
    Version 2: Advanced Edit Instruction Rewriter with categorized task types.
    Focuses on Subject, Appearance, Scene, and Advanced editing categories.
    """
    EDIT_SYSTEM_PROMPT_V2 = '''
# Advanced Edit Instruction Rewriter (Version 2)
You are a professional edit instruction rewriter. Your task is to generate precise, clear, and visually achievable professional-level edit instructions based on user-provided instructions and the image to be edited.

## Core Principles
- Maintain the **core intention** of the original instruction while enhancing clarity and feasibility
- Provide **sufficient detail** for unambiguous execution while staying concise
- Ensure all edits are **visually coherent** with the input image's style, lighting, and composition
- Resolve contradictions and supplement missing critical information intelligently

## Task Category Framework

### 1. SUBJECT Editing
**Scope:** Fundamental object-level manipulations - addition, removal, and replacement of subjects/objects in the image.

**Guidelines:**
- **Subject Addition:** Specify object type, position (foreground/midground/background, spatial location), size/scale, orientation, key visual attributes (color, texture, material), and interaction with existing elements
  > Example: "Add a brown leather armchair in the left foreground, angled 45° toward the window, with soft natural lighting creating subtle shadows"
  
- **Subject Removal:** Clearly identify the target object for removal; if multiple similar objects exist, specify which one(s)
  > Example: "Remove the red vase on the coffee table, preserving the wooden surface texture"
  
- **Subject Replacement:** Use format "Replace [object A] with [object B]" and describe key visual features of the new object to ensure style consistency
  > Example: "Replace the white ceramic lamp with a vintage brass lamp, maintaining warm lighting tone"

**Key Considerations:**
- Ensure spatial logic (perspective, occlusion, shadows)
- Match lighting conditions of the scene
- Maintain scale consistency with surroundings
- Consider depth relationships and layering

---

### 2. APPEARANCE Editing
**Scope:** Shape-preserving modifications to object appearance - color alteration, material modification, style transfer, and tone transformation.

**Guidelines:**
- **Color Alteration:** Specify target object and new color with descriptors (hue, saturation, value)
  > Example: "Change the blue sofa to emerald green, maintaining fabric texture and shadow gradients"
  
- **Material Modification:** Transform object material while preserving shape and form
  > Example: "Change the wooden table to polished marble with white-gray veining, keeping reflections and ambient lighting"
  
- **Style Transfer:** Apply artistic or photographic style while maintaining content
  > Example: "Transform to oil painting style: visible brushstrokes, rich color saturation, textured canvas appearance, impressionist technique"
  
- **Tone Transformation:** Adjust color temperature, mood, or atmospheric qualities
  > Example: "Shift to warm golden hour tones: increase orange-amber hues, soften shadows, add nostalgic warmth"

**Key Considerations:**
- Preserve object geometry and structure
- Maintain consistent lighting and shadows after appearance changes
- Ensure material properties are physically plausible (reflectivity, transparency, texture)
- Keep overall image harmony and color balance

---

### 3. SCENE Editing
**Scope:** Image layout and environmental understanding - background change, background extraction, and contextual modifications.

**Guidelines:**
- **Background Change:** Specify new background environment while emphasizing subject preservation
  > Example: "Replace background with misty mountain landscape at sunrise, maintain subject sharpness and original lighting on figure, blend edges naturally"
  
- **Background Extract/Isolation:** Remove or simplify background to highlight subject
  > Example: "Extract subject and replace background with clean white studio backdrop, preserve all subject details including hair strands and edge definition"
  
- **Scene Context Modification:** Alter environmental elements (time of day, weather, season, location)
  > Example: "Transform to rainy evening scene: add rain streaks, wet surface reflections, darker overcast lighting, street lamp glow"

**Key Considerations:**
- **Always prioritize subject consistency** - explicitly state to maintain subject integrity
- Ensure lighting coherence between subject and new background
- Match perspective and scale relationships
- Blend boundaries naturally (edge feathering, color adaptation)
- Consider atmospheric effects (depth haze, aerial perspective)

---

### 4. ADVANCED Editing
**Scope:** Complex tasks requiring sophisticated reasoning - portrait beautification, text modification, motion effects, and hybrid edits combining multiple categories.

**Guidelines:**

- **Portrait Beautification:** Enhance facial features naturally without over-processing
  > Example: "Enhance portrait: smooth skin texture while preserving natural pores, brighten eyes subtly, define facial contours gently, maintain authentic expression and ethnicity characteristics, professional retouching quality"
  
- **Text Modification:** Handle text edits with precision
  - All text in `" "` quotes, preserve original language and capitalization
  - Template: `Replace "original text" to "new text"` or `Replace the [location] text to "new text"`
  > Example: "Replace the storefront sign text \"CAFE\" to \"BISTRO\", maintain brown vintage lettering style and weathered texture"
  
- **Motion Change:** Add or modify dynamic elements
  > Example: "Add motion blur to running dog, directional blur from left to right, sharp focus on face, blurred legs and tail to convey speed"
  
- **Hybrid Edits:** Combine multiple edit types cohesively
  > Example: "Add golden retriever in foreground (Subject) + change background to park setting (Scene) + apply warm sunset lighting (Appearance), ensure consistent lighting across all elements"

**Key Considerations:**
- For portraits: maintain identity, subtle enhancements only, natural results
- For text: verify readability, appropriate sizing, stylistic consistency
- For motion: physically plausible dynamics, appropriate blur direction/intensity
- For hybrid: sequence edits logically, ensure global coherence, check inter-dependencies

---

## Universal Requirements

**Spatial Specifications:**
- Use clear position descriptors: "top-left corner", "center foreground", "right edge midground", "background behind [object]"
- Include relative positioning when helpful: "next to the chair", "above the table", "between the windows"

**Visual Attribute Details:**
When supplementing vague instructions, include:
- Color (specific hues, not just "red" but "crimson red" or "burgundy")
- Size (relative or absolute: "small", "filling 30% of frame", "life-sized")
- Texture (smooth, rough, glossy, matte, weathered, pristine)
- Lighting interaction (shadows, highlights, reflections)
- Orientation and angle

**Style Vocabulary Examples:**
- Photography: portrait, landscape, macro, aerial, documentary, editorial, fashion, street
- Art: impressionist, surrealist, minimalist, abstract, photorealistic, watercolor, sketch
- Mood: dramatic, serene, energetic, melancholic, vibrant, muted, ethereal
- Era: vintage 1920s, mid-century modern, contemporary, futuristic, timeless

**Logic and Consistency:**
- Resolve contradictions by choosing the most reasonable interpretation
- If instruction is impossible, adjust to closest achievable version
- Maintain image's original quality level (don't add unrealistic perfection to casual photos)
- Respect physical laws (gravity, perspective, lighting physics)

---

## Output Format
Provide rewritten instruction in JSON format:

```json
{
   "Rewritten": "[Your clear, detailed, and actionable instruction here]"
}
```

**Quality Checklist Before Output:**
☐ Task category identified (Subject/Appearance/Scene/Advanced)
☐ Core user intent preserved
☐ Sufficient detail provided for unambiguous execution
☐ Spatial/visual specifications included where needed
☐ Style consistency with input image maintained
☐ Physically and visually plausible
☐ Concise but complete (avoid unnecessary verbosity)
'''
    old_prompt = prompt
    prompt = f"{EDIT_SYSTEM_PROMPT_V2}\n\nUser Input: {prompt}\n\nRewritten Prompt:"
    success = False
    while not success:
        try:
            response = edit_api(prompt, [img], openai_client, model)
            result = response.choices[0].message.content if hasattr(response, "choices") else str(response)
            if isinstance(result, str):
                result = result.replace('```json','')
                result = result.replace('```','')
                result = json.loads(result)
            else:
                result = json.loads(result)

            polished_prompt = result['Rewritten']
            polished_prompt = polished_prompt.strip()
            polished_prompt = polished_prompt.replace("\n", " ")
            success = True
        except Exception as e:
            print(f"[Warning] Error during API call: {e}")

    # print(f"[V2] rewrite \"{old_prompt}\" -> \"{polished_prompt}\"")
    return polished_prompt


def evaluate_and_rewrite_prompt(
    original_image_path, 
    original_prompt, 
    rewritten_prompt, 
    edited_image_path, 
    openai_client, 
    model
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
    
    success = False
    while not success:
        try:
            content = [
                {"type": "text", "text": EVALUATE_REWRITE_PROMPT},
                {"type": "text", "text": f"\n\n## Input Data:\n\n**Original User Instruction:** {original_prompt}\n\n**Rewritten Prompt Used:** {rewritten_prompt}\n\n**Images Order:** [Original Image, Edited Image]\n\nPlease evaluate:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(original_image_path)}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(edited_image_path)}"}},
            ]
            
            response = openai_client.chat.completions.create(
                model=model,
                stream=False,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
            
            result = response.choices[0].message.content if hasattr(response, "choices") else str(response)
            
            # Parse JSON response
            if isinstance(result, str):
                result = result.replace('```json', '')
                result = result.replace('```', '')
                result = json.loads(result)
            else:
                result = json.loads(result)
            
            # Validate required fields
            if 'is_satisfied' not in result or 'reason' not in result:
                raise ValueError("Response missing required fields")
            
            success = True
            
            # Return result without printing (for cleaner logs)
            return result
            
        except Exception as e:
            print(f"[Warning] Error during evaluation API call: {e}")


def evaluate_and_rewrite_prompt_v2(
    original_image_path, 
    original_prompt, 
    rewritten_prompt, 
    edited_image_path, 
    openai_client, 
    model
):
    """
    Enhanced version 2 of evaluate_and_rewrite_prompt with improved rewriting rules.
    This version incorporates detailed task-specific guidelines from polish_edit_prompt
    for more stable and effective prompt refinement.
    
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
    EVALUATE_REWRITE_PROMPT_V2 = '''
# Edit Evaluation and Enhanced Prompt Refinement System (Version 2)

You are an expert image editing evaluator and prompt engineer. Your task is to:
1. Evaluate whether an edited image successfully fulfills the original user instruction
2. If not satisfied, generate an improved rewritten prompt following professional editing guidelines

## Input Information
You will receive:
- **Original Image**: The input image before editing
- **Original User Instruction**: The user's initial editing request
- **Rewritten Prompt**: The refined instruction that was used for editing
- **Edited Image**: The resulting image after applying the rewritten prompt

---

## PART 1: Evaluation Criteria

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

### Evaluation Decision

**SATISFIED**: The edited image successfully fulfills the original instruction with acceptable quality.
- Minor imperfections are acceptable if the core intent is met
- Use this when the edit is "good enough" for the user's purpose

**NOT SATISFIED**: The edited image fails to meet the original instruction in significant ways.
- Major elements are missing or incorrect
- Quality issues severely impact the result
- The rewritten prompt needs refinement

---

## PART 2: Enhanced Prompt Refinement Rules (if NOT SATISFIED)

When the result is NOT SATISFIED, you must generate a new rewritten prompt following these comprehensive professional editing guidelines:

### 1. General Principles for Rewriting
- Keep the rewritten prompt **concise**. Avoid overly long sentences and reduce unnecessary descriptive language.
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.
- Keep the core intention of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.
- All added objects or modifications must align with the logic and style of the edited input image's overall scene.

### 2. Task-Specific Rewriting Rules

#### 2.1 Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.
- If the description is vague, supplement with minimal but sufficient details:
  * **Category**: Specific object type (e.g., "brown leather armchair" not just "chair")
  * **Position**: Clear spatial location (e.g., "bottom-right corner", "left foreground near window")
  * **Size/Scale**: Relative or absolute size indicators
  * **Orientation**: Angle and facing direction
  * **Visual Attributes**: Color, texture, material, lighting interaction
  * **Interaction**: How it relates to existing elements
  
  > Example: 
  > Vague: "Add an animal"
  > Refined: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera, with natural lighting creating soft shadows"

- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.
- For replacement tasks, use format "Replace [Y] with [X]" and describe key visual features of X to ensure consistency.
  > Example: "Replace the white ceramic lamp with a vintage brass lamp, maintaining warm lighting tone and similar height"

#### 2.2 Text Editing Tasks
- **All text content must be enclosed in English double quotes `" "`**
- **Do not translate or alter the original language of the text, and do not change the capitalization**
- **For text replacement tasks, always use the fixed template:**
  * `Replace "xx" to "yy"`
  * `Replace the [location] text to "yy"`
  
- If the user does not specify text content, infer and add concise text based on the instruction and the input image's context:
  > Example:
  > Original: "Add a line of text" (on a poster)
  > Refined: "Add text \"LIMITED EDITION\" at the top center with slight shadow, using bold sans-serif font"

- Specify text position, color, font style, and layout in a concise way.

#### 2.3 Human/Portrait Editing Tasks
- **Maintain the person's core visual consistency**: ethnicity, gender, age, hairstyle, expression, outfit, etc.
- If modifying appearance (e.g., clothes, hairstyle), ensure the new element is consistent with the original style.
- **For expression changes, they must be natural and subtle, never exaggerated.**
- **If deletion is not specifically emphasized, the most important subject in the original image (e.g., a person, an animal) should be preserved.**
- For background change tasks, emphasize maintaining subject consistency at first.

  > Example:
  > Original: "Change the person's hat"
  > Refined: "Replace the man's hat with a dark brown beret; keep his smile, short hair, and gray jacket unchanged"

#### 2.4 Background/Scene Change Tasks
- **Always explicitly state to preserve the subject** when changing backgrounds
- Specify the new background environment clearly
- Ensure lighting coherence between subject and new background
- Address edge blending and natural transitions

  > Example:
  > Original: "Change background to beach"
  > Refined: "Replace background with sunny beach scene featuring sand, ocean waves, and blue sky, while strictly maintaining the subject's sharpness, facial features, hair details, and edge definition from the original image"

#### 2.5 Style Transformation or Enhancement Tasks
- If a style is specified, describe it concisely with key visual traits:
  > Example:
  > Original: "Disco style"
  > Refined: "1970s disco style: flashing lights, disco ball, mirrored walls, colorful tones, vibrant atmosphere"

- If the instruction says "use reference style" or "keep current style," analyze the input image, extract main features (color, composition, texture, lighting, art style), and integrate them into the prompt.

- **For coloring tasks, including restoring old photos, always use the fixed template:**
  "Restore old photograph, remove scratches, reduce noise, enhance details, high resolution, realistic, natural skin tones, clear facial features, no distortion, vintage photo restoration"

- If there are other changes, place the style description at the end.

### 3. Refinement Strategy Based on Failure Analysis

Analyze what went wrong and apply appropriate fixes:

**If the previous prompt was too vague:**
- Add more specific descriptors (exact colors, positions, sizes)
- Include spatial relationships and context
- Specify interaction with existing elements
- Add visual attributes (texture, lighting, shadows)

**If the previous prompt was contradictory:**
- Resolve conflicts between requirements
- Prioritize core intent over secondary details
- Simplify complex multi-part instructions
- Break down into clear sequential steps if needed

**If important details were lost:**
- Explicitly state preservation requirements using "maintain [aspect]" or "preserve [feature]"
- Reference specific elements from the original image that must be kept
- Add detail preservation clauses (e.g., "keep facial features unchanged")

**If positioning/scale was wrong:**
- Use more precise spatial descriptors (foreground/midground/background)
- Add relative size/scale indicators
- Include reference to nearby objects for spatial context
- Specify exact locations (top-left, center, bottom-right, etc.)

**If style/appearance was incorrect:**
- Use more specific visual vocabulary
- Add reference to original image's style elements
- Include material/texture/lighting specifications
- Specify color with more precision (e.g., "crimson red" not just "red")

**If the edit was over/under-processed:**
- Add degree modifiers: "subtle", "gentle", "moderate", "dramatic", "significant"
- Specify degree of change more clearly
- Balance enhancement with naturalness
- Set expectations for processing level

### 4. Rationality and Logic Checks
- Resolve contradictory instructions logically
- Add missing key information based on composition analysis
- Choose reasonable defaults for unspecified attributes
- Ensure physical plausibility (gravity, perspective, lighting physics)
- Maintain image's original quality level

---

## Output Format

```json
{
    "is_satisfied": true/false,
    "reason": "Detailed explanation of evaluation. If satisfied, explain why it meets requirements and what aspects are well-executed. If not satisfied, provide specific analysis of shortcomings: what went wrong, which elements are missing/incorrect, and what aspects need improvement.",
    "new_rewritten_prompt": "Only include if is_satisfied is false. The improved rewritten prompt following ALL the task-specific rules above. Must be precise, concise, and actionable. If is_satisfied is true, set this to null."
}
```

---

## Examples

### Example 1: Satisfied Result
```json
{
    "is_satisfied": true,
    "reason": "The edited image successfully adds a gray tabby cat in the bottom-right corner as requested. The cat is appropriately sized, naturally lit, and well-integrated into the scene with proper shadows. The cat's position and orientation (facing camera) are correct. All original elements are preserved without alteration. Minor shadow artifacts are barely noticeable and do not detract from the overall quality.",
    "new_rewritten_prompt": null
}
```

### Example 2: Not Satisfied - Vague Object Description
Original: "Add an animal"
Rewritten: "Add an animal in the corner"
Issue: Generated animal is wrong type, wrong size, poorly placed

```json
{
    "is_satisfied": false,
    "reason": "The edited image attempted to add an animal but the result is problematic: (1) The animal type is unclear and doesn't fit the scene context, (2) The size is disproportionate to the room, (3) The position 'in the corner' was too vague - it was placed awkwardly overlapping furniture. The rewritten prompt lacked specific details about animal type, exact position, size, orientation, and visual attributes.",
    "new_rewritten_prompt": "Add a medium-sized orange tabby cat in the bottom-right corner of the room, sitting on the floor near the sofa leg, facing toward the camera at a slight angle, with natural lighting from the window creating soft shadows on its fur"
}
```

### Example 3: Not Satisfied - Lost Subject Details (Background Change)
Original: "Change background to forest"
Rewritten: "Replace background with forest scene"
Issue: Subject's face became blurry, hair details lost

```json
{
    "is_satisfied": false,
    "reason": "While the background was successfully changed to a forest scene with trees and foliage, the edit severely compromised the subject quality: (1) Facial features became blurry and lost definition, (2) Hair details and edge sharpness were degraded, (3) Color bleeding occurred at the subject's edges. The rewritten prompt failed to explicitly emphasize subject preservation, which is critical for background change tasks.",
    "new_rewritten_prompt": "Replace background with lush forest scene featuring tall trees, green foliage, and natural lighting, while strictly maintaining the subject's sharpness, facial features, skin texture, hair details, and crisp edge definition from the original image, ensure no color bleeding at edges"
}
```

---

Now evaluate the provided images and prompts, and return your analysis in the specified JSON format.
'''
    
    success = False
    while not success:
        try:
            content = [
                {"type": "text", "text": EVALUATE_REWRITE_PROMPT_V2},
                {"type": "text", "text": f"\n\n## Input Data:\n\n**Original User Instruction:** {original_prompt}\n\n**Rewritten Prompt Used:** {rewritten_prompt}\n\n**Images Order:** [Original Image, Edited Image]\n\nPlease evaluate:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(original_image_path)}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(edited_image_path)}"}},
            ]
            
            response = openai_client.chat.completions.create(
                model=model,
                stream=False,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
            
            result = response.choices[0].message.content if hasattr(response, "choices") else str(response)
            
            # Parse JSON response
            if isinstance(result, str):
                result = result.replace('```json', '')
                result = result.replace('```', '')
                result = json.loads(result)
            else:
                result = json.loads(result)
            
            # Validate required fields
            if 'is_satisfied' not in result or 'reason' not in result:
                raise ValueError("Response missing required fields")
            
            success = True
            
            # Return result
            return result
            
        except Exception as e:
            print(f"[Warning] Error during evaluation API call (v2): {e}")


def polish_edit_prompt_with_negative(prompt, img, openai_client, model):
    """
    Enhanced version of polish_edit_prompt that also generates a negative prompt.
    
    Args:
        prompt: Original user instruction
        img: Path to the image to be edited
        openai_client: OpenAI client for API calls
        model: Model name to use
    
    Returns:
        dict: {
            'positive_prompt': str,  # Enhanced positive instruction
            'negative_prompt': str   # Negative prompt to avoid unwanted effects
        }
    """
    EDIT_SYSTEM_PROMPT_WITH_NEG = '''
# Edit Instruction Rewriter with Negative Prompt Generation

You are a professional edit instruction rewriter. Your task is to generate:
1. A precise, concise, and visually achievable positive edit instruction
2. A corresponding negative prompt to avoid common editing artifacts and unwanted effects

## PART 1: Positive Prompt Rewriting Rules

Please strictly follow the rewriting rules below for the positive prompt:

### 1. General Principles
- Keep the rewritten prompt **concise**. Avoid overly long sentences and reduce unnecessary descriptive language.  
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.  
- Keep the core intention of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.  
- All added objects or modifications must align with the logic and style of the edited input image's overall scene.  

### 2. Task Type Handling Rules

#### 2.1 Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.  
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:  
    > Original: "Add an animal"  
    > Rewritten: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"  
- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.  
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.  

#### 2.2 Text Editing Tasks
- All text content must be enclosed in English double quotes `" "`. Do not translate or alter the original language of the text, and do not change the capitalization.  
- **For text replacement tasks, always use the fixed template:**
    - `Replace "xx" to "yy"`.  
    - `Replace the xx bounding box to "yy"`.  
- If the user does not specify text content, infer and add concise text based on the instruction and the input image's context. For example:  
    > Original: "Add a line of text" (poster)  
    > Rewritten: "Add text \"LIMITED EDITION\" at the top center with slight shadow"  
- Specify text position, color, and layout in a concise way.  

#### 2.3 Human Editing Tasks
- Maintain the person's core visual consistency (ethnicity, gender, age, hairstyle, expression, outfit, etc.).  
- If modifying appearance (e.g., clothes, hairstyle), ensure the new element is consistent with the original style.  
- **For expression changes, they must be natural and subtle, never exaggerated.**  
- If deletion is not specifically emphasized, the most important subject in the original image (e.g., a person, an animal) should be preserved.
    - For background change tasks, emphasize maintaining subject consistency at first.  
- Example:  
    > Original: "Change the person's hat"  
    > Rewritten: "Replace the man's hat with a dark brown beret; keep smile, short hair, and gray jacket unchanged"  

#### 2.4 Style Transformation or Enhancement Tasks
- If a style is specified, describe it concisely with key visual traits. For example:  
    > Original: "Disco style"  
    > Rewritten: "1970s disco: flashing lights, disco ball, mirrored walls, colorful tones"  
- If the instruction says "use reference style" or "keep current style," analyze the input image, extract main features (color, composition, texture, lighting, art style), and integrate them into the prompt.  
- **For coloring tasks, including restoring old photos, always use the fixed template:** "Restore old photograph, remove scratches, reduce noise, enhance details, high resolution, realistic, natural skin tones, clear facial features, no distortion, vintage photo restoration"  
- If there are other changes, place the style description at the end.

### 3. Rationality and Logic Checks
- Resolve contradictory instructions: e.g., "Remove all trees but keep all trees" should be logically corrected.  
- Add missing key information: if position is unspecified, choose a reasonable area based on composition (near subject, empty space, center/edges).  

## PART 2: Negative Prompt Generation Rules

Generate a negative prompt that helps avoid:

### Common Issues to Avoid:
- **Artifacts & Distortions**: distortion, blurriness, noise, artifacts, deformation, warping
- **Quality Issues**: low quality, low resolution, poor details, oversaturation, undersaturation, washed out
- **Unwanted Changes**: unintended color shifts, loss of important details, wrong elements added
- **Visual Inconsistencies**: inconsistent lighting, unnatural shadows, incorrect perspective, scale mismatch
- **Blending Problems**: hard edges, color bleeding, halo effects, edge artifacts

### Task-Specific Negative Keywords:

**For Subject Addition:**
- floating objects, overlapping, wrong scale, disproportionate, no shadows, transparent, multiple copies, deformed body parts

**For Color/Material Changes:**
- unnatural colors, flat appearance, no texture, color bleeding, wrong saturation, loss of original details

**For Background Changes:**
- altered subject, blurry subject, changed clothing, face distortion, subject deformation, inconsistent lighting on subject

**For Text Editing:**
- blurry text, misspellings, wrong fonts, unreadable, distorted letters, incorrect text placement

**For Portrait/Human Editing:**
- distorted faces, unnatural expressions, identity loss, wrong ethnicity, age change, extra limbs, deformed features

**For Style Transfer:**
- inconsistent style application, partial transformation, conflicting styles, over-processed

The negative prompt should be concise, specific to the task, and use comma-separated keywords/phrases.

## Output Format

```json
{
   "positive_prompt": "Your enhanced positive instruction following the detailed rewriting rules above",
   "negative_prompt": "Comma-separated negative keywords specific to this task and image context"
}
```

## Examples

### Example 1: Color Change
Original: "change the car color to red"
```json
{
   "positive_prompt": "Change the car body color to bright red, maintaining metallic finish and reflections, preserve all other elements unchanged",
   "negative_prompt": "unnatural red, color bleeding, oversaturation, pink tint, orange tint, loss of details, flat color, no reflections, changed background, altered wheels, matte finish"
}
```

### Example 2: Object Addition
Original: "add a cat"
```json
{
   "positive_prompt": "Add a small gray tabby cat in the bottom-right corner, sitting naturally and facing the camera, with appropriate lighting and shadows",
   "negative_prompt": "floating cat, distorted proportions, wrong scale, blurry edges, no shadows, multiple cats, transparent cat, deformed body, overlapping objects"
}
```

### Example 3: Background Change
Original: "change background to beach"
```json
{
   "positive_prompt": "Replace background with sunny beach scene featuring sand, ocean waves, and blue sky, while strictly maintaining the subject's sharpness, details, and original lighting",
   "negative_prompt": "altered subject, blurry subject, changed clothing, face distortion, edge artifacts, color bleeding, inconsistent lighting, floating subject, halo effect, subject deformation"
}
```

### Example 4: Text Replacement
Original: "change the sign text to OPEN"
```json
{
   "positive_prompt": "Replace the storefront sign text to \"OPEN\", maintaining the original font style, color, and size",
   "negative_prompt": "blurry text, misspelled, wrong font, distorted letters, unreadable, incorrect capitalization, changed sign background, text too small, text too large"
}
```

Now process the user's instruction:
'''
    
    prompt_text = f"{EDIT_SYSTEM_PROMPT_WITH_NEG}\n\nUser Input: {prompt}\n\nGenerate both positive and negative prompts:"
    success = False
    
    while not success:
        try:
            response = edit_api(prompt_text, [img], openai_client, model)
            result = response.choices[0].message.content if hasattr(response, "choices") else str(response)
            
            if isinstance(result, str):
                result = result.replace('```json', '')
                result = result.replace('```', '')
                result = json.loads(result)
            else:
                result = json.loads(result)
            
            positive_prompt = result['positive_prompt'].strip().replace("\n", " ")
            negative_prompt = result['negative_prompt'].strip().replace("\n", " ")
            
            success = True
            return {
                'positive_prompt': positive_prompt,
                'negative_prompt': negative_prompt
            }
            
        except Exception as e:
            print(f"[Warning] Error during API call: {e}")
            # Return default if fails


def evaluate_and_rewrite_prompt_with_score(
    original_image_path, 
    original_prompt, 
    rewritten_prompt, 
    edited_image_path, 
    openai_client, 
    model,
    previous_negative_prompt=None
):
    """
    Enhanced version of evaluate_and_rewrite_prompt with scoring capability.
    Evaluates the edited image and provides:
    1. Satisfaction status
    2. Numerical scores for different aspects
    3. Refined prompts (positive and negative) if needed
    
    Args:
        original_image_path: Path to the original input image
        original_prompt: The original user instruction
        rewritten_prompt: The previously rewritten prompt that was used
        edited_image_path: Path to the edited image generated from rewritten_prompt
        openai_client: OpenAI client for API calls
        model: Model name to use
        previous_negative_prompt: The negative prompt that was used previously (optional)
    
    Returns:
        dict: {
            'is_satisfied': bool,
            'scores': {
                'overall': float,  # 0-10 overall quality score
                'alignment': float,  # 0-10 how well it matches the instruction
                'quality': float,  # 0-10 technical quality
                'preservation': float  # 0-10 preservation of important elements
            },
            'reason': str,
            'positive_prompt': str or None,
            'negative_prompt': str or None
        }
    """
    EVALUATE_SCORE_PROMPT = '''
# Edit Evaluation with Scoring and Prompt Refinement

You are an expert image editing evaluator and prompt engineer. Your task is to:
1. Evaluate whether an edited image successfully fulfills the original user instruction
2. Provide detailed numerical scores (0-10 scale)
3. If not satisfied, generate improved rewritten prompts (positive and negative) that address the shortcomings

## Input Information
You will receive:
- **Original Image**: The input image before editing
- **Original User Instruction**: The user's initial editing request
- **Rewritten Prompt Used**: The refined positive instruction that was used for editing
- **Previous Negative Prompt Used**: The negative prompt that was used (if any) to avoid unwanted effects
- **Edited Image**: The resulting image after applying the rewritten prompt

## PART 1: Evaluation Criteria

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

## PART 2: Scoring Criteria (0-10 scale, where 10 is perfect)

### 1. Overall Score (0-10)
Comprehensive assessment of the edit result
- 9-10: Excellent, meets all requirements perfectly
- 7-8: Good, minor imperfections but acceptable
- 5-6: Fair, noticeable issues but core goal achieved
- 3-4: Poor, significant problems
- 0-2: Failed, unusable result

### 2. Alignment Score (0-10)
How well the edited image matches the original instruction
- Are all requested changes present?
- Is the interpretation of the instruction correct?
- Are there any missing or extra modifications?

### 3. Quality Score (0-10)
Technical quality of the edited image
- Visual coherence and naturalness
- Absence of artifacts, distortions, or blurriness
- Proper lighting and shadows
- Clean edges and blending

### 4. Preservation Score (0-10)
How well important elements are maintained
- Subject integrity (if subject should be preserved)
- Important details retained
- No unintended changes to non-target areas
- Style consistency with original

## PART 3: Evaluation Decision

**SATISFIED**: The edited image successfully fulfills the original instruction with acceptable quality.
- Overall score >= 7.0
- Minor imperfections are acceptable if the core intent is met
- Use this when the edit is "good enough" for the user's purpose

**NOT SATISFIED**: The edited image fails to meet the original instruction in significant ways.
- Overall score < 7.0
- Major elements are missing or incorrect
- Quality issues severely impact the result
- The rewritten prompt needs refinement

## PART 4: Prompt Refinement Strategy (if NOT SATISFIED)

When generating new rewritten prompts (overall score < 7.0), analyze:

### 1. What went wrong?
- Compare original instruction → rewritten prompt → edited result
- Identify gaps between intent and execution
- Determine if the issue is clarity, specificity, or contradiction

### 2. Positive Prompt Refinement Approaches:

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

### 3. Leverage All Information:
- Reference what's visible in the original image
- Learn from what the previous rewritten prompt missed
- Use the edited image as feedback on what went wrong
- Maintain what worked, fix what didn't

### 4. Negative Prompt Generation:
- If a previous negative prompt was provided, analyze what it failed to prevent
- List specific artifacts or issues observed in the edited image
- Include general quality control keywords (blurriness, distortion, artifacts, etc.)
- Add task-specific negative keywords based on the failure patterns identified
- Keep effective keywords from the previous negative prompt and add new ones for observed issues
- Remove keywords from previous negative prompt that are irrelevant or ineffective
- Use comma-separated format for clarity

## Output Format

```json
{
    "is_satisfied": true/false,
    "scores": {
        "overall": 8.5,
        "alignment": 9.0,
        "quality": 8.0,
        "preservation": 9.0
    },
    "reason": "Detailed explanation of evaluation and scores. If satisfied, explain why it meets requirements. If not satisfied, explain specific shortcomings and what went wrong.",
    "positive_prompt": "Only include if is_satisfied is false. The improved positive rewritten prompt that addresses the identified issues. If is_satisfied is true, set this to null.",
    "negative_prompt": "Only include if is_satisfied is false. The negative prompt with comma-separated keywords to avoid the observed issues. If is_satisfied is true, set this to null."
}
```

## Examples

### Example 1: Satisfied Result
```json
{
    "is_satisfied": true,
    "scores": {
        "overall": 8.5,
        "alignment": 9.0,
        "quality": 8.5,
        "preservation": 8.0
    },
    "reason": "The edited image successfully adds a cat in the bottom-right corner as requested. The cat is appropriately sized, naturally lit, and well-integrated into the scene. Minor shadow artifacts are present but do not detract from the overall quality. All original elements preserved well.",
    "positive_prompt": null,
    "negative_prompt": null
}
```

### Example 2: Not Satisfied - Need More Specificity
Original: "Change the color"
Rewritten: "Change the object color to blue"
Previous Negative Prompt: "blurry, low quality, distorted"
Issue: Wrong object was recolored

```json
{
    "is_satisfied": false,
    "scores": {
        "overall": 5.0,
        "alignment": 4.0,
        "quality": 6.0,
        "preservation": 5.0
    },
    "reason": "The rewritten prompt was too vague about which object to recolor. The background was changed to blue instead of the intended subject (the car). The prompt needs to explicitly specify the target object. Quality is acceptable but alignment is poor due to wrong target. The previous negative prompt didn't address the specific issue of wrong target selection.",
    "positive_prompt": "Change the car color to blue, maintaining the metallic finish and reflections, keep all other elements including background unchanged",
    "negative_prompt": "changed background, wrong object recolored, flat color, color bleeding, loss of metallic finish, altered wheels, blurry, distorted"
}
```

### Example 3: Not Satisfied - Lost Important Details
Original: "Change background to beach"
Rewritten: "Replace background with beach scene"
Previous Negative Prompt: "color bleeding, inconsistent lighting, floating subject"
Issue: Subject was altered/degraded

```json
{
    "is_satisfied": false,
    "scores": {
        "overall": 5.5,
        "alignment": 7.0,
        "quality": 4.0,
        "preservation": 3.0
    },
    "reason": "While the background was changed to a beach scene (alignment: 7.0), the subject (person) lost facial detail and edge definition became blurry. The rewritten prompt failed to emphasize subject preservation. This severely impacts quality (4.0) and preservation (3.0) scores. The previous negative prompt mentioned 'inconsistent lighting' and 'floating subject' but missed critical subject preservation issues like blurriness and face distortion.",
    "positive_prompt": "Replace background with sunny beach scene featuring sand, ocean, and clear sky, while strictly maintaining the subject's sharpness, facial features, hair details, and edge definition from the original image",
    "negative_prompt": "blurred subject, face distortion, loss of facial details, soft edges, subject alteration, changed clothing, edge artifacts, halo effect, inconsistent lighting on subject, floating subject, color bleeding"
}
```

### Example 4: Not Satisfied - Positioning Error
Original: "Add a lamp"
Rewritten: "Add a lamp"
Previous Negative Prompt: "floating objects, wrong scale, no shadows"
Issue: Lamp was added in awkward position

```json
{
    "is_satisfied": false,
    "scores": {
        "overall": 6.0,
        "alignment": 6.0,
        "quality": 7.0,
        "preservation": 8.0
    },
    "reason": "A lamp was added as requested but placed in the center of the floor, which looks unnatural and awkward. The rewritten prompt lacked spatial guidance. Quality and preservation are good, but the poor positioning brings down the alignment and overall scores. The previous negative prompt prevented 'floating objects' and 'wrong scale' issues successfully, but didn't address positioning problems.",
    "positive_prompt": "Add a modern floor lamp in the left corner near the sofa, approximately 5-6 feet tall, with warm lighting that complements the room's ambiance",
    "negative_prompt": "center placement, awkward positioning, floating lamp, no base, wrong scale, inconsistent lighting, out of place, no shadows"
}
```

Now evaluate the provided images and prompts and return your analysis in the specified JSON format:
'''
    
    success = False
    while not success:
        try:
            # Build input data text
            input_data_text = f"\n\n## Input Data:\n\n**Original User Instruction:** {original_prompt}\n\n**Rewritten Prompt Used:** {rewritten_prompt}\n\n"
            
            if previous_negative_prompt:
                input_data_text += f"**Previous Negative Prompt Used:** {previous_negative_prompt}\n\n"
            else:
                input_data_text += f"**Previous Negative Prompt Used:** None (no negative prompt was used)\n\n"
            
            input_data_text += "**Images Order:** [Original Image, Edited Image]\n\nPlease evaluate with scores:"
            
            content = [
                {"type": "text", "text": EVALUATE_SCORE_PROMPT},
                {"type": "text", "text": input_data_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(original_image_path)}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(edited_image_path)}"}},
            ]
            
            response = openai_client.chat.completions.create(
                model=model,
                stream=False,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
            
            result = response.choices[0].message.content if hasattr(response, "choices") else str(response)
            
            # Parse JSON response
            if isinstance(result, str):
                result = result.replace('```json', '')
                result = result.replace('```', '')
                result = json.loads(result)
            else:
                result = json.loads(result)
            
            # Validate required fields
            required_fields = ['is_satisfied', 'scores', 'reason']
            if not all(field in result for field in required_fields):
                raise ValueError("Response missing required fields")
            
            # Validate scores structure
            required_scores = ['overall', 'alignment', 'quality', 'preservation']
            if not all(score in result['scores'] for score in required_scores):
                raise ValueError("Response missing required score fields")
            
            success = True
            return result
            
        except Exception as e:
            print(f"[Warning] Error during evaluation API call: {e}")
            # Return a safe default

if __name__ == "__main__":
    import openai
    # Please replace with your own API key
    openai.api_key = "your_api_key_here"  # Replace with your OpenAI API key
    # Please replace with your own API endpoint
    base_url = "https://your-endpoint.com/v1/openai/native"  # Replace with your API endpoint
    api_key = openai.api_key
    # model = "gpt-4o-2024-11-20"
    model = "qwen-vl-max-0813"
    openai_client = openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    # Please replace with your own image path
    ins = polish_edit_prompt("change rabbit to blue", '/path/to/your/image.jpg', openai_client, model)  # Replace with your image path
    print(ins)