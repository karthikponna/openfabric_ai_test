import json
import re
import ollama
from typing import Dict, List, Optional

from logger.logging import logger

# System prompt
INTENT_ANALYZER_SYSTEM_PROMPT = """
You are an AI assistant for an art generation system. Your job is to look at what the user is asking for and decide if they are talking about something from a previous conversation that happened before this current chat session.

You will get the current chat session history as context. Use this to understand what has already been created or discussed in this current session.

Current Session Context: {currentSessionContext}

**Your Task:**
Look at the user's request and decide:
- If they are asking about something from the CURRENT session (shown above), return false because we don't need to search old memories
- If they are asking about something from a PREVIOUS session (not in current context), return true because we need to search old memories
- If it's a completely new request, return false

**Examples of when to return true (need old memories):**
- "create a robot like the one I made last week"
- "remake the dragon from yesterday" 
- "add wings to the castle I created before"
- "make another version of my old car design"

**Examples of when to return false (don't need old memories):**
- "create a new robot" (completely new)
- "make it red" (referring to something in current session)
- "add wings to it" (referring to current session item)
- "change the color" (referring to current session item)

**Response Format:**
You must respond with ONLY a JSON object like this:
{{"requiresMemory": true}} or {{"requiresMemory": false}}

Do not include any other text or explanations.

**Example Response 1:**
{{"requiresMemory": true}}

**Example Response 2:** 
{{"requiresMemory": false}}
"""

# User prompt
USER_PROMPT = "User's request: {userPrompt}"


def check_for_memory_intent(
    user_prompt: str,
    current_session_history: Optional[List[Dict[str, str]]] = None
) -> bool:
    """
    Uses an LLM to determine if a user's prompt requires long-term memory access.

    Args:
        user_prompt: The user's latest input string.
        current_session_history: The current session conversation history.

    Returns:
        True if memory is required, False otherwise.
    """
    current_session_history = current_session_history or []
    
    # Filtering to get only text entries from the session history
    text_entries = [msg for msg in current_session_history if isinstance(msg.get('content'), str)]

    if text_entries:
        context_lines = [f"{msg.get('role', 'unknown')}: {msg.get('content')}" for msg in text_entries]
        session_context = "\n".join(context_lines)
        logger.info(f"[Intent Analyzer] Current session context: {session_context}")

    else:
        session_context = "No previous conversation in this session."
    
    try:
        
        formatted_system_prompt = INTENT_ANALYZER_SYSTEM_PROMPT.format(
            currentSessionContext=session_context
        )
        
        formatted_user_prompt = USER_PROMPT.format(userPrompt=user_prompt)
        
        response = ollama.chat(
            model="deepseek-r1:14b",
            messages=[
                {"role": "system", "content": formatted_system_prompt},
                {"role": "user", "content": formatted_user_prompt}
            ],
            options={"temperature": 0.0} # We want a non creative response
        )
        response_content = response['message']['content'].strip()
        
        logger.info(f"[Intent Analyzer] Raw response: '{response_content}'")

        # Remove <think> tags and any other reasoning wrapper tags
        response_cleaned = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
        

        try:
            response_json = json.loads(response_cleaned)
            requires_memory = response_json.get("requiresMemory", False)
            logger.info(f"[Intent Analyzer] Parsed JSON response: {requires_memory}")
        except json.JSONDecodeError:
            
            # Look for JSON object in the response
            json_match = re.search(r'\{[^}]*"requiresMemory"[^}]*\}', response_content)
            if json_match:
                json_str = json_match.group(0)
                response_json = json.loads(json_str)
                requires_memory = response_json.get("requiresMemory", "")

            else:
                requires_memory = False
                logger.warning(f"[Intent Analyzer] Could not parse response, defaulting to False: '{response_content}'")

        logger.info(f"Intent Analyzer for prompt '{user_prompt}': requires_memory={requires_memory}")
        return requires_memory

    except Exception as e:
        logger.error(f"[Intent Analyzer] Failed to analyze intent: {e}", exc_info=True)
        return False