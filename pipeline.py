import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512

LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
        prompt,
        uncond_prompt=None,
        input_image=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=50,
        models={},
        seed=None, 
        device=None,
        idle_device=None,
        tokenizer=None
    ):
    
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("Strength must be between 0 and 1.")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)

        else:
            to_idle = lambda x:x
            
        generator = torch.Generator(device=device)

        if seed is None:
            generator.seed()

        else:
            generator.manual_seed(seed)

        clip = models["clip"]

        clip.to(device)

        if do_cfg:

            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding = "max_length", max_length = 77
            ).input_ids
