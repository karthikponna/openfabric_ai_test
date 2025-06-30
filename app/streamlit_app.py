import base64
import uuid
import re
import json
import streamlit as st
import streamlit.components.v1 as components

from logger.logging import logger
from utils import load_json

from core.stub import Stub
from src.llm import enhance_prompt
from src.user_intent_llm import check_for_memory_intent
from database.memory_manager import find_similar_prompts, save_generation

st.set_page_config(layout="wide", page_title="AI Developer Challenge")

def load_app_ids():
    """
    Loads Openfabric application IDs using the utility loader.
 
    Returns:
        A list of application IDs, or an empty list on error.
    """
    try:
        state = load_json("config/state.json")
        return state.get("super-user", {}).get("app_ids", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Could not load or parse config/state.json: {e}")
        st.error(f"Could not load or parse config/state.json: {e}")
        return []

def render_3d_model(model_bytes):
    """Renders a 3D model using the model-viewer component."""
    b64_model = base64.b64encode(model_bytes).decode("utf-8")
    model_viewer_html = f"""
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
        <model-viewer style="width: 100%; height: 400px;" src="data:model/gltf-binary;base64,{b64_model}"
        ar ar-modes="webxr scene-viewer quick-look" camera-controls tone-mapping="neutral"
        poster="https://placehold.co/600x400/eee/eee?text=Loading..." shadow-intensity="1"
        environment-image="neutral" auto-rotate>
        </model-viewer>
    """
    components.html(model_viewer_html, height=400)


# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"New session started: {st.session_state.session_id}")

if "history" not in st.session_state:
    st.session_state.history = [] # Will store dicts of {'type': 'text/imag/3d/, 'role': 'user'/'assistant', 'content':...}

# Main app UI
try:
    with open("assets/openfabric_logo.png", "rb") as img_file:
        img_bytes = img_file.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        st.markdown(f"""
            <div style='text-align: center; margin-bottom: 20px;'>
                <h1 style='margin: 0; font-size: 3rem;'>🚀 AI Creative Partner</h1>
                <div style='display: flex; align-items: center; justify-content: center; margin-top: 2px; margin-left: 500px;'>
                    <span style='font-size: 18px; color: #666; margin-right: 8px;'>powered by</span>
                    <img src='data:image/png;base64,{img_b64}' width='70' height='70'>
                </div>
            </div>
        """, unsafe_allow_html=True)
except FileNotFoundError:
    # Fallback if logo file is not found
    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1 style='margin: 0; font-size: 3rem;'>🚀 AI Creative Partner</h1>
            <div style='margin-top: 10px; margin-left: 200px; font-size: 16px; color: #666;'>powered by Openfabric</div>
        </div>
    """, unsafe_allow_html=True)

# Display past messages from history
for entry in st.session_state.history:
    if entry['type'] == 'text':
        with st.chat_message(entry['role']):
            st.markdown(entry['content'])

    elif entry['type'] == 'image':
        # display historical image
        with st.chat_message('assistant'):
            st.image(entry['content'], width=400)

    elif entry['type'] == '3d':
        # render historical 3d model
        with st.chat_message('assistant'):
            render_3d_model(entry['content'])

# Main chat input
if prompt := st.chat_input("Describe what you want to create..."):

    # 1. Add user message to history and display it
    st.session_state.history.append({'type': 'text', 'role': 'user', 'content': prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Start the AI's response process
    with st.chat_message("assistant"):
        try:
            # Meory retrieval logic
            retrieved_memory = None

            # updating the text history to include only text messages
            text_history = [m for m in st.session_state.history if m['type'] == 'text']
            with st.spinner("🧠 Analyzing your intent..."):
                requires_memory = check_for_memory_intent(prompt, text_history)

            if requires_memory:
                with st.spinner("🧠 Accessing long-term memories..."):
                    similar = find_similar_prompts(prompt, k=1)
                    if similar:
                        retrieved_memory = similar[0]
                        st.info(f"Found a related memory from {retrieved_memory['timestamp']}:\n> {retrieved_memory['enhanced_prompt']}")
            

            with st.spinner("🎨 Enhancing your idea..."):
                # We pass the user prompt, current session history, and long-term memory to the LLM
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
                        # If JSON doesn't have expected key, using the whole cleaned response
                        enhanced_prompt = enhanced_response

                except json.JSONDecodeError:
                    # If it's not JSON, use the cleaned response as it is
                    enhanced_prompt = enhanced_response

                # Save the *enhanced* prompt to the history for short-term memory
                st.session_state.history.append({'type': 'text', 'role': 'assistant', 'content': f"**Enhanced Prompt:** {enhanced_prompt}"})
                st.markdown(f"**Enhanced Prompt:** {enhanced_prompt}")

            # Openfabric app IDs loading
            app_ids = load_app_ids()
            if not app_ids or len(app_ids) < 2:
                st.error("Two Openfabric app IDs are required. Check your config.")
                st.stop()

            stub = Stub(app_ids)

            # Call Text-to-Image App
            with st.spinner("🖼️ Generating image..."):
                resp_img = stub.call(app_ids[0], {"prompt": enhanced_prompt}, uid="super-user")
            img_bytes = resp_img.get("result") 

            if not img_bytes:
                st.error("Failed to generate image. The response was empty.")
                st.stop()

            # Display image and add to history
            st.image(img_bytes, width=400)
            st.session_state.history.append({'type': 'image', 'role': 'assistant', 'content': img_bytes})

            # Call Image-to-3D App
            with st.spinner("🧊 Generating 3D model... (this can take a moment)"):
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                resp_3d = stub.call(app_ids[1], {"input_image": img_b64}, 'super-user')
                model_bytes = resp_3d.get('generated_object')

            if not model_bytes:
                st.warning("3D model generation finished, but no model data was returned.")
                st.stop()

            # Display 3D model and add to history
            render_3d_model(model_bytes)
            st.session_state.history.append({'type': '3d', 'role': 'assistant', 'content': model_bytes})

            # 3. Save the enhanced prompt and user prompt to long-term memory
            with st.spinner("💾 Saving to long-term memory..."):
                save_generation(
                    session_id=st.session_state.session_id,
                    user_prompt=prompt,
                    enhanced_prompt=enhanced_prompt
                )
            st.success("Creation saved to long-term memory!")

        except Exception as e:
            logger.error(f"An error occurred in the main pipeline: {e}", exc_info=True)
            st.error(f"An unexpected error occurred: {e}")