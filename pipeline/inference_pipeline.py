import argparse
import gc
import json
import os
import random
import re
import sys
import time
from glob import iglob
from pathlib import Path
from typing import Optional, Dict


import lpips
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image
from transformers import T5EncoderModel, T5Tokenizer


MAX_SEED = np.iinfo(np.int32).max
current_file_path = Path(__file__).resolve() # Ensure correct sys path setup for local imports
sys.path.insert(0, str(current_file_path.parent.parent))


from cache_merge import patch
from diffusion import IDDPM, DPMS, SASolverSampler
from diffusion.data.datasets.utils import *
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from diffusion.model.utils import prepare_prompt_ar, resize_and_crop_tensor
from diffusion.utils.dist_utils import flush
from diffusion.utils.logger import get_root_logger
from diffusers.models import AutoencoderKL
from prompt import prompts
from tools.download import find_model


def get_args():
    parser = argparse.ArgumentParser()
    # == Model configuration == #
    parser.add_argument('--image_size', default=2048, type=int)
    parser.add_argument('--version', default='sigma', type=str)
    parser.add_argument('--model_path', default='output/pretrained_models/PixArt-Sigma-XL-2-2K-MS.pth', type=str)
    parser.add_argument('--sdvae', action='store_true', help='sd vae')

    # == Sampling configuration == #
    parser.add_argument('--seed', default=4242424242, type=int, help='Seed for the random generator')
    parser.add_argument('--sampler', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--sample_steps', default=20, type=int, help='Number of inference steps')
    parser.add_argument('--guidance_scale', default=5.0, type=float, help='Guidance scale')

    # ==== Acceleration Patch ==== #
    parser.add_argument('--experiment-folder', type=str, default='samples/inference/cfg=6.0')
    parser.add_argument("--evaluate-lpips", action=argparse.BooleanOptionalAction, type=bool, default=False)

    # ==== 1. Merging ==== #
    parser.add_argument("--merge-ratio", type=float, default=0.0)
    parser.add_argument("--merge-metric", type=str, choices=["k", "x"], default="k")
    parser.add_argument("--merge-mode", type=str, choices=["token_merge", "cache_merge"], default="cache_merge")
    parser.add_argument("--prune", action=argparse.BooleanOptionalAction, type=bool, default=True)
    parser.add_argument("--merge-cond", action=argparse.BooleanOptionalAction, type=bool, default=False) # only use when you got a low cfg (3.0-4.5)

    parser.add_argument('--merge-path', type=str, default=None)
    parser.add_argument("--merge-step", type=lambda s: (int(item) for item in s.split(',')), default=(0, 18))
    parser.add_argument("--cache-step", type=lambda s: (int(item) for item in s.split(',')), default=(7, 18))
    parser.add_argument("--push-unmerged", action=argparse.BooleanOptionalAction, type=bool, default=True)

    # ==== 2. Broadcasting ==== #
    parser.add_argument("--broadcast-range", type=int, default=1, help="broadcast range, set 1 to bypass")
    parser.add_argument("--broadcast-step", type=lambda s: (int(item) for item in s.split(',')), default=(7, 18))
    parser.add_argument("--broadcast-start-blocks", type=lambda s: [int(item) for item in s.split(',')], default=[1])
    parser.add_argument("--broadcast-num-blocks", type=lambda s: [int(item) for item in s.split(',')], default=[28])

    # ==== 3. Other ==== #
    parser.add_argument('--other-arg', type=str, help='An argument needed for experiment')

    args, unknown = parser.parse_known_args()
    return args

def set_env(args, seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def load_merge_path_from_file(file_path: str) -> Dict[int, list]:
    try:
        with open(file_path, 'r') as f:
            serializable_paths = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error reading or parsing merge path file: {e}")

    return {
        description: {int(block): steps for block, steps in path.items()}
        for description, path in serializable_paths.items()
    }


def get_merge_path(args: argparse.Namespace, merge_path: Optional[Dict[int, list]] = None) -> Dict[int, list]:
    if merge_path is not None:
        return merge_path

    if args.merge_path:
        paths = load_merge_path_from_file(args.merge_path)
        first_description = next(iter(paths))
        return paths[first_description]

    return {i: [] for i in range(1, 29)}


def main(prompts, merge_ratio = None, merge_path = None):
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_root_logger()

    # parse argument
    if merge_ratio is not None:
        args.merge_ratio = merge_ratio

    merge_path = get_merge_path(args, merge_path)

    # Validate image size
    valid_image_sizes = [256, 512, 1024, 2048]
    assert args.image_size in valid_image_sizes, (
        "We only provide pre-trained models for 256x256, 512x512, 1024x1024, and 2048x2048 resolutions."
    )

    # Initialize variables
    pe_interpolation = {256: 0.5, 512: 1, 1024: 2, 2048: 4}
    latent_size = args.image_size // 8
    max_sequence_length = {"alpha": 120, "sigma": 300}[args.version]
    weight_dtype = torch.float16
    micro_condition = args.version == 'alpha' and args.image_size == 1024

    # Initialize the model
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

    # Load model weights
    state_dict = find_model(args.model_path)
    state_dict = state_dict.get('state_dict', {})
    state_dict.pop('pos_embed', None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.warning(f'Missing keys: {missing}')
    logger.warning(f'Unexpected keys: {unexpected}')

    model.to(dtype=weight_dtype)

    # Initialize components
    vae = AutoencoderKL.from_pretrained(
        "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="vae"
    ).to(device, dtype=weight_dtype)
    tokenizer = T5Tokenizer.from_pretrained(
        "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="tokenizer"
    )
    text_encoder = T5EncoderModel.from_pretrained(
        "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", load_in_8bit=True, subfolder="text_encoder"
    )
    null_caption_token = tokenizer(
        "", max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
    ).to(device)
    null_caption_embs = text_encoder(
        null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask
    )[0]

    # Apply acceleration patch if needed
    if args.merge_ratio > 0.0:
        model = patch.apply_patch(
            model,
            metric=args.merge_metric,
            ratio=args.merge_ratio,
            mode=args.merge_mode,
            prune=args.prune,
            sx=2,
            sy=2,
            latent_size=latent_size,
            merge_cond=args.merge_cond,
            merge_path=merge_path,
            merge_step=args.merge_step,
            cache_step=args.cache_step,
            push_unmerged=args.push_unmerged,
            broadcast_range=args.broadcast_range,
            broadcast_step=args.broadcast_step,
            broadcast_start_blocks=args.broadcast_start_blocks,
            broadcast_num_blocks=args.broadcast_num_blocks
        )

    model.eval()
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

    @torch.inference_mode()
    def generate_img(prompt, sampler, sample_steps, scale, seed=0, randomize_seed=False):
        flush()
        gc.collect()
        torch.cuda.empty_cache()

        seed = int(randomize_seed_fn(seed, randomize_seed))
        set_env(args, seed)

        logger.info(prompt)
        prompt_clean, prompt_show, hw, ar, custom_hw = prepare_prompt_ar(
            prompt, base_ratios, device=device
        )
        prompt_clean = prompt_clean.strip()
        prompts_list = [prompt_clean]

        # Tokenize and encode prompt
        caption_token = tokenizer(
            prompts_list,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        caption_embs = text_encoder(
            caption_token.input_ids, attention_mask=caption_token.attention_mask
        )[0]
        emb_masks = caption_token.attention_mask

        caption_embs = caption_embs[:, None]
        null_y = null_caption_embs.repeat(len(prompts_list), 1, 1)[:, None]

        latent_size_h = int(hw[0, 0] // 8)
        latent_size_w = int(hw[0, 1] // 8)

        # Sample images based on the sampler
        n = len(prompts_list)
        model_kwargs = {'data_info': {'img_hw': hw, 'aspect_ratio': ar}, 'mask': emb_masks}
        start_time = time.perf_counter()

        if sampler == 'iddpm':
            z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
            model_kwargs.update({
                'y': torch.cat([caption_embs, null_y]),
                'cfg_scale': scale
            })
            diffusion = IDDPM(str(sample_steps))
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                device=device
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        elif sampler == 'dpm-solver':
            z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
            dpm_solver = DPMS(
                model.forward_with_dpmsolver,
                condition=caption_embs,
                uncondition=null_y,
                cfg_scale=scale,
                model_kwargs=model_kwargs
            )
            samples = dpm_solver.sample(
                z,
                steps=sample_steps,
                order=2,
                skip_type="time_uniform",
                method="multistep"
            )

        elif sampler == 'sa-solver':
            sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
            samples = sa_solver.sample(
                S=sample_steps,
                batch_size=n,
                shape=(4, latent_size_h, latent_size_w),
                eta=1,
                conditioning=caption_embs,
                unconditional_conditioning=null_y,
                unconditional_guidance_scale=scale,
                model_kwargs=model_kwargs,
            )[0]

        else:
            raise RuntimeError("Invalid scheduler")

        runtime = time.perf_counter() - start_time

        # Decode samples
        samples = samples.to(dtype=weight_dtype)
        samples = vae.decode(samples / vae.config.scaling_factor).sample
        samples = resize_and_crop_tensor(samples, custom_hw[0, 1], custom_hw[0, 0])

        return samples.squeeze(0), runtime

    outputs = []
    total_runtime = 0.0
    os.makedirs(args.experiment_folder, exist_ok=True)

    # Generate images
    for prompt in prompts:
        output, runtime = generate_img(
            prompt,
            sampler=args.sampler,
            sample_steps=args.sample_steps,
            scale=args.guidance_scale,
            seed=args.seed,
            randomize_seed=False
        )
        outputs.append(output)
        total_runtime += runtime
        patch.reset_cache(model)

    outputs = torch.stack(outputs, dim=0)

    # Save generated images
    output_name_pattern = os.path.join(args.experiment_folder, "img_{idx}.jpg")
    if not os.path.exists(args.experiment_folder):
        os.makedirs(args.experiment_folder)
        idx = 0
    else:
        existing_files = [
            fn for fn in iglob(output_name_pattern.format(idx="*"))
            if re.search(r"img_[0-9]+\.jpg$", fn)
        ]
        idx = (
            max(int(fn.split("_")[-1].split(".")[0]) for fn in existing_files) + 1
            if existing_files else 0
        )

    if args.merge_ratio == 0.0:
        fn = f"{args.experiment_folder}/baseline.png"
    else:
        fn = output_name_pattern.format(idx=idx)

    save_image(outputs, fn, nrow=4, normalize=True, value_range=(-1, 1))
    logger.info(f"Finished sampling {len(prompts)} images in {total_runtime:.2f} seconds.")
    logger.info(f"Saved generated image at {fn}")

    # Evaluate LPIPS if required
    if args.evaluate_lpips and args.merge_ratio > 0.0:
        baseline_path = f"{args.experiment_folder}/baseline.png"
        if not os.path.exists(baseline_path):
            logger.info("Baseline image does not exist!")
        else:
            logger.info("Evaluating LPIPS")
            p_r = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])(Image.open(baseline_path).convert('RGB')).to(device)
            p_o = outputs.to(device)
            loss_fn_alex = lpips.LPIPS(net='alex').to(device)
            d = loss_fn_alex(p_r, p_o)
            logger.info(f"LPIPS: {d.item()}")
            return d.item()

    return None


if __name__ == '__main__':
    paths = load_merge_path_from_file('path_12.json')
    first_description = next(iter(paths))
    merge_path = paths[first_description]

    main(prompts, merge_path=merge_path)