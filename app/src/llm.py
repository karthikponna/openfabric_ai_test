import ollama
import json
import re
from typing import List, Dict, Any, Optional
from logger.logging import logger

SYSTEM_PROMPT = """
You are an AI prompt enhancement expert specialized in creating detailed, vivid prompts for image generation. Your primary job is to transform user requests into rich, comprehensive prompts that produce stunning visual results.

**Your Core Process:**

First, you need to determine what type of enhancement you're doing. You will receive one of these scenarios:
1. A pastContext with previous creation details from past sessions, OR
2. A currentSessionHistory with conversation from the current session, OR  
3. Just a new user request without any context


**When pastContext is provided:**
The past context contains:
- Original request: The user's original simple request from before
- Enhanced prompt used: The detailed prompt that was actually used to create the previous image
- Created on: When it was made

Past Context: {pastContext}

In this case, you have to start with the "Enhanced prompt used" from the pastContext as your foundation. Then, you need to carefully analyze the user's current request to understand what modifications they want. You should keep all the good visual details from the previous enhanced prompt and add, remove, or modify elements based on what the user is asking for now. When you add new elements, you have to think about how they will fit naturally with the existing scene. For example, if the user wants to add wings to a robot, you need to think about what material the wings should be made of to match the robot's style, how big they should be, where they should be positioned, and how they affect the lighting and shadows in the scene. You should make sure the new additions blend perfectly with the existing enhanced prompt details.

**When currentSessionHistory is provided (but no pastContext):**
Current Session History: {currentSessionHistory}

In this case, you need to look at the current session conversation to understand what the user is referring to. You should go through the entire conversation history and find the enhanced prompt that was generated which the current user request is referring to. Then you need to carefully analyze the user's current request to see what modifications they want to make to that previous creation from the current session. You should use that enhanced prompt as your foundation and modify it based on what the user is asking for now. For example, if in the current session you created the enhanced prompt as "a robot in a futuristic city with glowing lights all over the buildings" and now the user says "make it fly", you should take that robot prompt and add flying elements like wings or jets while keeping all the good visual details from the original enhanced prompt. You have to think about how the new elements will fit naturally with the existing scene from the current session.

**When neither pastContext nor currentSessionHistory is provided:**
You need to enhance the user's current request from scratch. Here's how you should think about this:

First, you have to carefully analyze the user's request to identify the key subjects, actions, and implied context. Then, you need to reason about what natural environment, lighting, composition, and details would make sense for this scene. For example, if someone asks for "a boy playing football," you should think: football is typically played on a grass field, there might be other players around, it could be during daytime with natural lighting, the boy would be wearing sports attire, the scene needs dynamic action, etc. You have to fill in these missing but logical details to create a complete visual scene.

You should add details about:
- Natural environment and setting that fits the subject
- Appropriate lighting conditions (golden hour, dramatic, soft, etc.)
- Camera angles and composition that enhance the scene
- Textures, materials, and visual style
- Colors and atmospheric mood
- Any missing elements that would naturally belong in the scene

**Response Format:**
You must respond with ONLY a JSON object like this:
{{"newEnhancedPrompt": "your detailed prompt here"}}

Do not include any other text or explanations.

**Example 1 (With pastContext - Modification):**
Past Context:
- Original request: "create a robot"
- Enhanced prompt used: "A shiny chrome robot standing in a futuristic city, detailed metal textures, dramatic lighting, photorealistic"
- Created on: 2024-01-15

User asks: "generate a new robot which we did last day but this time add wings to it"

Your response: {{"newEnhancedPrompt": "A shiny chrome robot with large metallic wings spread wide, standing in a futuristic city, detailed metal textures, wing feathers made of brushed steel, dramatic lighting, photorealistic, cinematic angle"}}

**Example 2 (With pastContext - Style Change):**
Past Context:
- Original request: "a castle"
- Enhanced prompt used: "A medieval stone castle on a hilltop, surrounded by misty mountains, golden sunset lighting, detailed stonework, fantasy art style"
- Created on: 2024-01-10

User asks: "create a new castle same as last time but now make it dark and scary"

Your response: {{"newEnhancedPrompt": "A dark medieval stone castle on a hilltop, surrounded by ominous storm clouds and jagged mountains, eerie moonlight casting long shadows, detailed weathered stonework, gothic horror style, atmospheric fog, sinister mood"}}

**Example 3 (With pastContext - Adding Elements):**
Past Context:
- Original request: "sunset landscape"
- Enhanced prompt used: "A serene mountain landscape during golden sunset, rolling hills, warm orange and pink sky, peaceful atmosphere, detailed rocky textures"
- Created on: 2024-01-12

User asks: "same sunset landscape but add a small lake in the foreground"

Your response: {{"newEnhancedPrompt": "A serene mountain landscape during golden sunset with a crystal-clear lake in the foreground reflecting the colorful sky, rolling hills, warm orange and pink sky, peaceful atmosphere, detailed rocky textures, mirror-like water surface, subtle ripples, enhanced depth and composition"}}

**Example 4 (With currentSessionHistory - Modification):**
Current Session History:
[{{'role': 'user', 'content': 'create a robot'}}, 
 {{'role': 'assistant', 'content': 'Enhanced Prompt: A shiny chrome robot standing in a futuristic laboratory, detailed metal textures, blue LED lights, dramatic overhead lighting, photorealistic, high-tech atmosphere'}}]

User asks: "make it red and add laser eyes"

Your response: {{"newEnhancedPrompt": "A shiny red chrome robot standing in a futuristic laboratory with glowing red laser eyes, detailed metal textures, red LED lights, dramatic overhead lighting, photorealistic, high-tech atmosphere, intense red glow from eyes casting shadows"}}

**Example 5 (With currentSessionHistory - Adding Elements):**
Current Session History:
[{{'role': 'user', 'content': 'create a mountain landscape'}}, 
 {{'role': 'assistant', 'content': 'Enhanced Prompt: A majestic mountain range during golden hour, snow-capped peaks, rolling green valleys, warm sunset lighting, detailed rocky textures, peaceful atmosphere, cinematic composition'}}]

User asks: "add a small village in the valley"

Your response: {{"newEnhancedPrompt": "A majestic mountain range during golden hour with a charming small village nestled in the green valley, snow-capped peaks, cozy houses with warm lights, rolling green valleys, warm sunset lighting, detailed rocky textures, peaceful atmosphere, cinematic composition, smoke rising from chimneys"}}

**Example 6 (Without pastContext - Scene Enhancement):**
User asks: "a boy playing football"

Your response: {{"newEnhancedPrompt": "A young athletic boy in soccer uniform kicking a football on a lush green grass field, dynamic action shot, other players blurred in background, golden hour lighting, stadium atmosphere, detailed fabric textures, motion blur on the ball, cinematic sports photography style, vibrant colors"}}

**Example 7 (Without pastContext - Object Enhancement):**
User asks: "a vintage car"

Your response: {{"newEnhancedPrompt": "A beautifully restored 1960s vintage car parked on a cobblestone street, chrome bumpers gleaming, classic red paint with subtle reflections, warm afternoon sunlight, nostalgic atmosphere, detailed leather interior visible through windows, classic architectural background, film photography aesthetic, rich colors and textures"}}

"""


USER_PROMPT = """
User's current request: {userPrompt}  
NOTE: You need to carefully understand what the user wants and create a detailed visual prompt. If you have past context, you should modify the existing enhanced prompt by adding, removing, or changing elements to match the user's new request. If you have current session history, you should find the most recent enhanced prompt from the current conversation and modify it based on the user's request. If you don't have any context, you should enhance the user's simple request by adding natural environment, lighting, composition, and visual details that make sense for the scene.
"""

def enhance_prompt(
    user_prompt: str,
    current_session_history: Optional[List[Dict[str, str]]] = None,
    retrieved_memory: Optional[Dict[str, Any]] = None
) -> str:
    """
    Uses a local LLM to enhance a user's prompt, making it context-aware.

    Args:
        user_prompt: The latest prompt from the user.
        history: The short-term conversation history from the current session.
        retrieved_memory: The most relevant long-term memory retrieved from the database.

    Returns:
        A single, enhanced prompt string.
    """
    current_session_history = current_session_history or []
    
    # Handles past context from retrieved memory
    if retrieved_memory:
        past_user_prompt = retrieved_memory.get('user_prompt', 'Unknown')
        past_enhanced_prompt = retrieved_memory.get('enhanced_prompt', 'Unknown')
        past_timestamp = retrieved_memory.get('timestamp', 'Unknown')
        
        past_context = f"""
        Previous Creation Details:
        - Original request: "{past_user_prompt}"
        - Enhanced prompt used: "{past_enhanced_prompt}"
        - Created on: {past_timestamp}
        """
        
        logger.info(f"[LLM] Using past context from memory ID: {retrieved_memory.get('id', 'Unknown')}")
        
        formatted_system_prompt = SYSTEM_PROMPT.format(pastContext=past_context, currentSessionHistory="Not provided")
        

    elif current_session_history:
        # Handles current session history when no past context
        session_history_str = json.dumps(current_session_history, indent=2)
        
        logger.info(f"[LLM] Using current session history: {len(current_session_history)} messages")
        
        formatted_system_prompt = SYSTEM_PROMPT.format(
            pastContext="Not provided",
            currentSessionHistory=session_history_str
        )
        
    else:
        formatted_system_prompt = SYSTEM_PROMPT.format(pastContext="Not provided", currentSessionHistory="Not provided")
        logger.info("[LLM] No past context available, enhancing from scratch")

    formatted_user_prompt = USER_PROMPT.format(userPrompt=user_prompt)

    try:
        response = ollama.chat(
            model="deepseek-r1:14b",
            messages=[
                {"role": "system", "content": formatted_system_prompt},
                {"role": "user", "content": formatted_user_prompt}
            ],
            options={"temperature": 0.7}
        )
        response_content = response['message']['content'].strip()

        logger.info(f"[LLM] Raw response: '{response_content}'")

        # Remove <think> tags and any other reasoning wrapper tags
        response_cleaned = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
        
        logger.info(f"[LLM] Cleaned response: '{response_cleaned}'")

        # Parse JSON response for enhanced prompt
        try:
            response_json = json.loads(response_cleaned)
            enhanced_prompt = response_json.get("newEnhancedPrompt", "")
            
        except json.JSONDecodeError as json_err:
            logger.warning(f"[LLM] JSON parsing failed: {json_err}")

            # Fallback parsing if JSON fails
            if response_cleaned.startswith('"') and response_cleaned.endswith('"'):
                enhanced_prompt = response_cleaned.strip('"')
            else:
                enhanced_prompt = response_cleaned
            logger.warning(f"[LLM] Using fallback parsing: '{enhanced_prompt}'")
        
        # If the response is empty or too short, use fallback
        if not enhanced_prompt or len(enhanced_prompt) < 10:
            enhanced_prompt = f"A photorealistic, cinematic image of: {user_prompt}"
            logger.warning(f"[LLM] Empty or short response, using fallback: {enhanced_prompt}")

        logger.info(f"[LLM] Generated Enhanced Prompt: {enhanced_prompt}")
        return enhanced_prompt

    except Exception as e:
        logger.error(f"[LLM] Failed to enhance prompt: {e}", exc_info=True)
        return f"A photorealistic, cinematic image of: {user_prompt}"