import torch
from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionPipeline

model_path = "sayakpaul/sd-model-finetuned-lora-t4"
noise_scheduler = GaudiDDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
pipe = GaudiStableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.bfloat16,
        use_habana=True,
        use_hpu_graphs=True,
        gaudi_config="Habana/stable-diffusion",
        scheduler=noise_scheduler,
)

pipe.unet.load_attn_procs(model_path)

prompt = "A pokemon with green eyes and red legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("pokemon.png")

