
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
import torch
import os 

model_dir = "checkpoints/ldmcheckpoint/ldm.pth"

def generate_image(user_input, pipe):
    # pipe = AutoPipelineForText2Image.from_pretrained(model_dir, torch_dtype=torch.float16, variant="fp16")
    # image = pipe(prompt=user_input).images[0]
    # torch.cuda.set_device(1)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(1)  # Set the device to GPU1 (replace with the correct GPU index)
    #     device = "cuda"
    # else:
    # device = "cpu"
        
    # pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16 if device == "cuda" else torch.float32, variant="fp16").to(device)
    # pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    
    # # unet_checkpoint = torch.load("/checkpoints/ldmcheckpoint.pth", map_location="cpu")
    # # unet_checkpoint = os.path.join("checkpoints", "ldmcheckpoint") 
    # unet_checkpoint = os.path.join(os.path.dirname(__file__), "checkpoints", "ldmcheckpoint", "ldm.pth")
    # if os.path.exists(unet_checkpoint):
    #     unet_checkpoint = torch.load(unet_checkpoint)
    #     pipe.unet.load_state_dict(unet_checkpoint, strict=False)
    # else:
    #     raise FileNotFoundError(f"Checkpoint not found at {unet_checkpoint}")
    
    # pipe.unet.load_state_dict(unet_checkpoint, strict=False)
    # # pipe.unet.to("cuda")
    generated_image = pipe(prompt=user_input, num_inference_steps=1, guidance_scale=0.0).images[0]
    # image = pipe(prompt=user_input, num_inference_steps=10, guidance_scale=0.0).images[0]

    return generated_image
