import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from calculate.utils import empty_directory

directory_path = "data/diffusion"
model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "arm | beak | bird | catch | cockatoo | hand | person | man | parrot | perch | pet | white | yellow"

empty_directory(directory_path)
for i in range(0,3):
    image = pipe(prompt=prompt).images[0]
    image.save(f"{directory_path}/result_{i}.png")