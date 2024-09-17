import argparse
import sys
import json
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import os
import random
import torch
from torchvision.utils import save_image
from diffusion import DPMS
from diffusers.models import AutoencoderKL
from tools.download import find_model
from datetime import datetime
from typing import List, Union
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
import gc
from tqdm import tqdm
from PIL import Image

from diffusion.model.t5 import T5Embedder
from diffusion.model.utils import prepare_prompt_ar, resize_and_crop_tensor
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
from torchvision.utils import _log_api_usage_once, make_grid
from diffusion.data.datasets.utils import *
from asset.examples import examples
from diffusion.utils.dist_utils import flush

from prompt import PROMPT
from cache_merge import patch


MAX_SEED = np.iinfo(np.int32).max


def get_args():
    parser = argparse.ArgumentParser()
    # == Model configuration == #
    parser.add_argument('--image_size', default=2048, type=int)
    parser.add_argument('--version', default='sigma', type=str)
    parser.add_argument('--model_path', default='output/pretrained_models/PixArt-Sigma-XL-2-2K-MS.pth', type=str)
    parser.add_argument('--sdvae', action='store_true', help='sd vae')

    # == Sampling configuration == #
    parser.add_argument('--seed', default=0, type=int, help='Seed for the random generator')
    parser.add_argument('--sample_steps', default=20, type=int, help='Number of inference steps')
    parser.add_argument('--guidance_scale', default=7.5, type=float, help='Guidance scale')

    # == Evaluation configuration == #
    parser.add_argument("--num-fid-samples", type=int, default=20)

    # ==== ==== ==== ==== ==== ==== ==== ==== ==== #
    # ==== Acceleration Patch ==== #
    parser.add_argument('--experiment-folder', type=str, default='samples/experiment/[YOUR-EXP-NAME]')

    # ==== 1. Merging ==== #
    parser.add_argument("--merge-ratio", type=float, default=0.4, help="Ratio of tokens to merge")
    parser.add_argument("--merge-metric", type=str, choices=["k", "x"], default="k")
    parser.add_argument("--merge-cond", action=argparse.BooleanOptionalAction, type=bool, default=False)

    # == 1.1 Token Merging (Spatial) == #
    parser.add_argument("--start-indices", type=lambda s: [int(item) for item in s.split(',')], default=[])
    parser.add_argument("--num-blocks", type=lambda s: [int(item) for item in s.split(',')], default=[])

    # == 1.2 Cache Merging (Spatial-Temporal) == #
    parser.add_argument("--cache-start-indices", type=lambda s: [int(item) for item in s.split(',')], default=[8, 21])
    parser.add_argument("--cache-num-blocks", type=lambda s: [int(item) for item in s.split(',')], default=[8, 2])
    parser.add_argument("--cache-step", type=lambda s: (int(item) for item in s.split(',')), default=(7, 18))
    parser.add_argument("--push-unmerged", action=argparse.BooleanOptionalAction, type=bool, default=True)

    # == 2. Broadcast (Temporal) == #
    parser.add_argument("--broadcast-range", type=int, default=2, help="broadcast range, set 0 to bypass")
    parser.add_argument("--broadcast-step", type=lambda s: (int(item) for item in s.split(',')), default=(7, 18))
    parser.add_argument("--broadcast-start-indices", type=lambda s: [int(item) for item in s.split(',')], default=[1, 16])
    parser.add_argument("--broadcast-num-blocks", type=lambda s: [int(item) for item in s.split(',')], default=[7, 5])

    return parser.parse_args()


def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


@torch.inference_mode()
def generate_img(prompt, sampler, sample_steps, scale, seed=0, randomize_seed=False):
    flush()
    gc.collect()
    torch.cuda.empty_cache()

    seed = int(randomize_seed_fn(seed, randomize_seed))
    set_env(seed)

    os.makedirs(f'output/demo/online_demo_prompts/', exist_ok=True)
    save_promt_path = f'output/demo/online_demo_prompts/tested_prompts{datetime.now().date()}.txt'

    with open(save_promt_path, 'a') as f:
        f.write(prompt + '\n')
    prompt_clean, prompt_show, hw, ar, custom_hw = prepare_prompt_ar(prompt, base_ratios, device=device)      # ar for aspect ratio
    prompt_clean = prompt_clean.strip()

    if isinstance(prompt_clean, str):
        prompts = [prompt_clean]

    caption_token = tokenizer(prompts, max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
    caption_embs = text_encoder(caption_token.input_ids, attention_mask=caption_token.attention_mask)[0]
    emb_masks = caption_token.attention_mask

    caption_embs = caption_embs[:, None]
    null_y = null_caption_embs.repeat(len(prompts), 1, 1)[:, None]

    latent_size_h, latent_size_w = int(hw[0, 0]//8), int(hw[0, 1]//8)

    # Sample images:
    if sampler == 'dpm-solver':
        # Create sampling noise:
        n = len(prompts)
        z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
        dpm_solver = DPMS(model.forward_with_dpmsolver,
                          condition=caption_embs,
                          uncondition=null_y,
                          cfg_scale=scale,
                          model_kwargs=model_kwargs)
        samples = dpm_solver.sample(
            z,
            steps=sample_steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )
    else:
        raise RuntimeError("Use only DPM-Solver for FID Eval")

    samples = samples.to(weight_dtype)
    samples = vae.decode(samples / vae.config.scaling_factor).sample

    samples = resize_and_crop_tensor(samples, custom_hw[0,1], custom_hw[0,0])
    return samples.squeeze(0)


if __name__ == '__main__':
    from diffusion.utils.logger import get_root_logger
    assert torch.cuda.is_available()

    args = get_args()
    device = 'cuda'
    logger = get_root_logger()

    assert args.image_size in [256, 512, 1024, 2048], \
        "We only provide pre-trained models for 256x256, 512x512, 1024x1024 and 2048x2048 resolutions."
    pe_interpolation = {256: 0.5, 512: 1, 1024: 2, 2048: 4}
    latent_size = args.image_size // 8
    max_sequence_length = {"alpha": 120, "sigma": 300}[args.version]
    weight_dtype = torch.float16
    micro_condition = True if args.version == 'alpha' and args.image_size == 1024 else False
    if args.image_size in [512, 1024, 2048]:
        model = PixArtMS_XL_2(
            input_size=latent_size,
            pe_interpolation=pe_interpolation[args.image_size],
            micro_condition=micro_condition,
            model_max_length=max_sequence_length,
        ).to(device)
    else:
        model = PixArt_XL_2(
            input_size=latent_size,
            pe_interpolation=pe_interpolation[args.image_size],
            model_max_length=max_sequence_length,
        ).to(device)
    state_dict = find_model(args.model_path)
    if 'pos_embed' in state_dict['state_dict']:
        del state_dict['state_dict']['pos_embed']
    missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    logger.warning(f'Missing keys: {missing}')
    logger.warning(f'Unexpected keys: {unexpected}')
    model.to(weight_dtype)

    if args.merge_ratio > 0.0:
        model = patch.apply_patch(model,
                                  merge_metric=args.merge_metric,
                                  ratio=args.merge_ratio,
                                  merge_cond=args.merge_cond,
                                  sx=1, sy=3, latent_size=latent_size,

                                  start_indices=args.start_indices,
                                  num_blocks=args.num_blocks,

                                  cache_start_indices=args.cache_start_indices,
                                  cache_num_blocks=args.cache_num_blocks,
                                  cache_step=args.cache_step,
                                  push_unmerged=args.push_unmerged,

                                  broadcast_range=args.broadcast_range,
                                  broadcast_step=args.broadcast_step,
                                  broadcast_start_indices=args.broadcast_start_indices,
                                  broadcast_num_blocks=args.broadcast_num_blocks)

    model.eval()
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

    if args.sdvae:
        vae = AutoencoderKL.from_pretrained("output/pretrained_models/sd-vae-ft-ema").to(device).to(weight_dtype)
    else:
        vae = AutoencoderKL.from_pretrained(f"PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="vae").to(device).to(weight_dtype)

    tokenizer = T5Tokenizer.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", load_in_8bit=True, subfolder="text_encoder")
    null_caption_token = tokenizer("", max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
    null_caption_embs = text_encoder(null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask)[0]

    # Prepare Prompts
    with open('pipeline/data/annotations/captions_val2014.json', 'r') as f:
        coco_data = json.load(f)

    annotations = coco_data['annotations']
    prompts = [ann['caption'] for ann in annotations]

    if len(prompts) > args.num_fid_samples:
        prompts = random.sample(prompts, args.num_fid_samples)

    # Sampling
    index = 0
    os.makedirs(args.experiment_folder, exist_ok=True)

    for prompt in prompts:
        output = generate_img(prompt, 'dpm-solver', args.sample_steps, args.guidance_scale, seed=args.seed, randomize_seed=False)
        save_path = f"{args.experiment_folder}/{index:06d}.png"
        save_image(output, save_path, nrow=4, normalize=True, value_range=(-1, 1))
        patch.reset_cache(model)
        index += 1

    create_npz_from_sample_folder(args.experiment_folder, args.num_fid_samples)
    print("Done.")