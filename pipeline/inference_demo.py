import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16

transformer = Transformer2DModel.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    subfolder='transformer',
    torch_dtype=weight_dtype,
    use_safetensors=True,
)
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    transformer=transformer,
    torch_dtype=weight_dtype,
    use_safetensors=True,
)
pipe.to(device)

# Enable memory optimizations.
# pipe.enable_model_cpu_offload()

prompt = "A huge cactus with a happy face in the Sahara desert"
image = pipe(prompt,
             height=1024,
             width=1024,
             guidance_scale=7.0
             ).images[0]
image.save("./hugetus.png")