import streamlit as st
from models.m2m import generate_text
from models.ldm import generate_image
from models.textstyle import generate_styled_image

from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
import torch
import os
import io

import time
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer
)

@st.cache_resource
def load_pipe():
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32, revision="fp16").to("cpu")
        
    unet_checkpoint = os.path.join(os.path.dirname(__file__), "models", "checkpoints", "ldmcheckpoint", "ldm.pth")
    if os.path.exists(unet_checkpoint):
        unet_checkpoint = torch.load(unet_checkpoint, map_location=torch.device('cpu'))
        pipe.unet.load_state_dict(unet_checkpoint, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {unet_checkpoint}")
    
    return pipe

@st.cache_resource
def load_m2m():
    model_dir = os.path.join(os.path.dirname(__file__), "models", "checkpoints", "m2mcheckpoint")
    model = M2M100ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = M2M100Tokenizer.from_pretrained(model_dir, src_lang="en", tgt_lang="ne")
    tokenizer.src_lang = "en"
    return model, tokenizer


pipe = load_pipe()
model, tokenizer = load_m2m()

# Streamlit app
def main():
    # Initialize session state if not already present
    if 'page' not in st.session_state:
        st.session_state.page = "Text Generation"  # Start with the Text Generation page
    
    # Based on current page, call the relevant function
    if st.session_state.page == "Text Generation":
        text_generation_page(model, tokenizer)
    elif st.session_state.page == "Image Generation":
        image_generation_page(pipe)
    elif st.session_state.page == "Style Generation":
        text_styling_page()

def text_generation_page(model, tokenizer):
    # Page 1: Text Generation
    st.title("Text Generation")
    
    # Input from the user
    english_prompt = st.text_area("Enter your English prompt:", key="english_prompt_input")
    
    if st.button("Generate Nepali Text"):
        if english_prompt:
            # Generate Nepali wish using the user input
            nepali_wish = generate_text(english_prompt, model, tokenizer)
            st.write("Generated Nepali Wish:")
            st.write(nepali_wish)
            st.session_state.nepali_wish = nepali_wish  # Store the generated wish for later use
            st.success("Text Generated! Click Next to go to Image Generation.")
            # Update the session state to move to the next page
            # st.session_state.page = "Image Generation"
        else:
            st.warning("Please enter a valid prompt.")


    # Rerun button to reset text and enter a new prompt
        if st.button("Rerun Text Generation"):
            del st.session_state.nepali_wish  # Remove stored text
            st.session_state.page = "Text Generation"
            nepali_wish = generate_text(english_prompt)
            st.write("Generated Nepali Wish:")
            st.write(nepali_wish)
            st.session_state.nepali_wish = nepali_wish  # Store the generated wish for later use
            st.success("Text Generated! Click Next to go to Image Generation.")
            st.rerun()  # Rerun to refresh UI

    # Next button to go to Image Generation
    if st.button("Next: Generate Image"):
        st.session_state.page = "Image Generation"
        # image_generation_page()
        st.rerun()

def image_generation_page(pipe):
    # Page 2: Image Generation
    st.title("Image Generation")
    
    if "nepali_wish" not in st.session_state:
        st.warning("Please generate text first before proceeding.")
        return
    
    # Input from the user for image generation
    image_prompt = st.text_area("Enter image description:", key="image_prompt_input")
    
    if st.button("Generate Image"):
        if image_prompt:
            # Generate image using the provided description
            generated_image = generate_image(image_prompt, pipe)
            st.image(generated_image, caption="Generated Image", use_column_width=True)
            st.session_state.generated_image = generated_image  # Store the generated image
            st.success("Image Generated! Click Next to go to Text Styling.")
            # Update the session state to move to the next page
            # st.session_state.page = "Text Styling"
        else:
            st.warning("Please enter a valid description.")
    
            if st.button("Rerun Image Generation"):
                del st.session_state.nepali_wish  # Remove stored text
                generated_image = generate_image(image_prompt)
                st.image(generated_image, caption="Generated Image", use_column_width=True)
                st.session_state.generated_image = generated_image  # Store the generated image
                st.success("Image Generated! Click Next to go to Text Styling.")
                st.rerun()  # Rerun to refresh UI

    # Next button to go to Image Generation
    if st.button("Next: Text Styling"):
        st.session_state.page = "Style Generation"
        # text_styling_page()
        st.rerun()

def text_styling_page():
    # Page 3: Text Styling and Integration
    st.title("Poster Generation")
    
    if "nepali_wish" not in st.session_state or "generated_image" not in st.session_state:
        st.warning("Please generate text and image first before proceeding.")
        return
    
    # Combine the generated text with the image
    if st.button("Generate Styled Image"):
        styled_image = generate_styled_image(st.session_state.nepali_wish, st.session_state.generated_image)
        st.image(styled_image, caption="Styled Image", use_column_width=True)
        st.success("Styled Image Generated!")
        
        img_buffer = io.BytesIO()
        styled_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        st.download_button(
            label="Download Styled Image",
            data=img_buffer,
            file_name="styled_image.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
