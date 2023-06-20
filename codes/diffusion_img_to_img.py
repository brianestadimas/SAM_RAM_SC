import torch
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from calculate.utils import empty_directory
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

directory_path = "data/diffusion"
model_id = "stabilityai/stable-diffusion-2"
device = "cuda"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
# # # Workaround for not accepting attention shape using VAE for Flash Attention
pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
pipe.enable_attention_slicing()

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
prompt = "arm | beak | bird | catch | cockatoo | hand | person | man | parrot | perch | pet | white | yellow"
# generator = torch.Generator(device=device).manual_seed(1024)

init_image = Image.open("data/bird.jpg").convert("RGB")
# init_image = Image.open(BytesIO(response.content)).convert("RGB")
# init_image.thumbnail((768, 768))

empty_directory(directory_path)
for i in range(0,3):
    image = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
    image.save(f"{directory_path}/result_{i}.png")