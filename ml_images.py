from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image

token_path = Path("token.txt")
token = token_path.read_text().strip()


################################ Model to generate image ################################
def model_to_generate_image():
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    return pipe

def obtain_image(prompt:str) -> Image:
    pipe = model_to_generate_image()
    image = pipe(prompt).images[0]
    return image

############################### NER ###############################
    

