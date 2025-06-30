import uuid
import base64
import json
import re
from typing import Dict

from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from openfabric_pysdk.context import AppModel, State
from core.stub import Stub

from logger.logging import logger
from src.llm import enhance_prompt
from src.user_intent_llm import check_for_memory_intent
from database.memory_manager import find_similar_prompts, save_generation

# Configurations for the app
configurations: Dict[str, ConfigClass] = dict()

############################################################
# Config callback function
############################################################
def config(configuration: Dict[str, ConfigClass], state: State) -> None:
    """
    Stores user-specific configuration data.

    Args:
        configuration (Dict[str, ConfigClass]): A mapping of user IDs to configuration objects.
        state (State): The current state of the application (not used in this implementation).
    """
    for uid, conf in configuration.items():
        logger.info(f"Saving new config for user with id:'{uid}'")
        configurations[uid] = conf


############################################################
# Execution callback function
############################################################
def execute(model: AppModel) -> None:
    """
    Main execution entry point for handling a model pass.

    Args:
        model (AppModel): The model object containing request and response structures.
    """

    logger.info("Starting execution workflow...")
    response: OutputClass = model.response
    
    try:
        # Retrieve input
        request: InputClass = model.request
        prompt: str = request.prompt
        if not prompt:
            response.message = "Error: Input prompt cannot be empty."
            logger.error(response.message)
            return

        logger.info(f"Received prompt: {prompt}")

        # Retrieve user config
        user_config: ConfigClass = configurations.get('super-user', None)
        if not user_config or not user_config.app_ids or len(user_config.app_ids) < 2:
            response.message = "Error: Configuration is missing or incomplete. Two app_ids are required."
            logger.error(response.message)
            return
            
        logger.info(f"Loaded user config with {len(user_config.app_ids)} app_ids.")

        # Initialize the Stub with app IDs
        app_ids = user_config.app_ids
        stub = Stub(app_ids)

        # ------------------------------
        # AI Generation Workflow
        # ------------------------------
        
        # A session ID is generated for each execution to track the process.
        session_id = str(uuid.uuid4())
        logger.info(f"Generated session ID: {session_id}")

        # Since each execution is stateless, we start with an empty history.
        text_history = [{'role': 'user', 'content': prompt}]

        # 1. Analyze user's intent to check if long-term memory is needed
        logger.info("Analyzing user intent for memory retrieval...")
        requires_memory = check_for_memory_intent(prompt, text_history)
        
        retrieved_memory = None

        if requires_memory:
            logger.info("Intent analysis suggests memory retrieval is required. Searching...")
            similar = find_similar_prompts(prompt, k=1)
            if similar:
                retrieved_memory = similar[0]
                logger.info(f"Found a related memory: {retrieved_memory['enhanced_prompt']}")

        else:
            logger.info("Intent analysis suggests no memory retrieval needed.")

        # 2. Enhance the user prompt
        logger.info("Enhancing user prompt...")
        enhanced_response = enhance_prompt(
            user_prompt=prompt,
            current_session_history=text_history,
            retrieved_memory=retrieved_memory
        )

        # Extract the enhanced prompt from JSON if it's in JSON format
        try:
            # Try to parse as JSON first
            enhanced_json = json.loads(enhanced_response)
            if isinstance(enhanced_json, dict) and "newEnhancedPrompt" in enhanced_json:
                enhanced_prompt = enhanced_json["newEnhancedPrompt"]

            else:
                # If JSON doesn't have expected key, use the whole cleaned response
                enhanced_prompt = enhanced_response
        except json.JSONDecodeError:
            # If it's not JSON, use the cleaned response as it is
            enhanced_prompt = enhanced_response

        logger.info(f"Enhanced prompt: {enhanced_prompt}")
        text_history.append({'role': 'assistant', 'content': f"**Enhanced Prompt:** {enhanced_prompt}"})

        # 3. Call Text-to-Image App
        logger.info(f"Calling Text-to-Image app (ID: {app_ids[0]})...")
        resp_img = stub.call(app_ids[0], {"prompt": enhanced_prompt}, uid="super-user")
        img_bytes = resp_img.get("result")

        if not img_bytes:
            response.message = "Error: Failed to generate image. The response was empty."
            logger.error(response.message)
            return
        
        logger.info("Image generation successful.")

        # 4. Call Image-to-3D App
        logger.info(f"Calling Image-to-3D app (ID: {app_ids[1]})...")
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        resp_3d = stub.call(app_ids[1], {"input_image": img_b64}, 'super-user')
        model_bytes = resp_3d.get('generated_object')

        if not model_bytes:
            # This is treated as a warning as the image was still generated.
            logger.warning("3D model generation finished, but no model data was returned.")
            response.message = f"Workflow partially completed. Image generated, but 3D model failed. Enhanced Prompt was: {enhanced_prompt}"
        else:
            logger.info("3D model generation successful.")
            response.message = f"Workflow completed successfully! Your enhanced prompt was: {enhanced_prompt}"

        # 5. Save the generation to long-term memory
        logger.info("Saving generation to long-term memory...")
        save_generation(
            session_id=session_id,
            user_prompt=prompt,
            enhanced_prompt=enhanced_prompt
        )
        
        logger.info("Successfully saved to long-term memory.")

    except Exception as e:
        logger.error(f"An unexpected error occurred in the execution workflow: {e}", exc_info=True)
        response.message = f"An unexpected error occurred: {e}"

