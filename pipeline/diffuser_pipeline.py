"""
Degraded! Fix later
"""

import time
import os
import argparse
import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline
from pytorch_lightning import seed_everything

from cache_merge import patch
from prompt import PROMPT
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    # == Model configuration == #
    parser.add_argument('--model_path', default='PixArt-alpha/PixArt-Sigma-XL-2-1024-MS', type=str)
    parser.add_argument('--seed', default=82, type=int, help='Seed for the random generator')
    parser.add_argument('--sample_steps', default=20, type=int, help='Number of inference steps')
    parser.add_argument('--guidance_scale', default=7.0, type=float, help='Guidance scale')

    # ==== ==== ==== ==== ==== ==== ==== ==== ==== #
    # ==== Token Merging Configuration ==== #
    parser.add_argument('--experiment-folder', type=str, default='samples/experiment/diffuser')
    parser.add_argument("--merge-ratio", type=float, default=0.5, help="Ratio of tokens to merge")
    parser.add_argument("--start-indices", type=lambda s: [int(item) for item in s.split(',')], default=[9, 21])
    parser.add_argument("--num-blocks", type=lambda s: [int(item) for item in s.split(',')], default=[8, 2])

    # == Improvements == #
    parser.add_argument("--unmerge-residual", action=argparse.BooleanOptionalAction, type=bool, default=True)
    parser.add_argument("--cache_step", type=lambda s: (int(item) for item in s.split(',')), default=(5, 15))
    parser.add_argument("--push-unmerged", action=argparse.BooleanOptionalAction, type=bool, default=True)

    # == Hybrid Unmerge == #
    parser.add_argument("--hybrid-unmerge", type=float, default=0.0,
                        help="cosine similarity threshold, set 0.0 to bypass")

    return parser.parse_args()


def save_grid(images, num_rows, output_path):
    num_images = len(images)
    num_cols = (num_images + num_rows - 1) // num_rows

    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)

    grid_width = num_cols * max_width
    grid_height = num_rows * max_height

    grid_image = Image.new('RGB', (grid_width, grid_height))

    for index, image in enumerate(images):
        row = index // num_cols
        col = index % num_cols
        x = col * max_width
        y = row * max_height
        grid_image.paste(image, (x, y))

    grid_image.save(output_path)


if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16
    seed_everything(args.seed)

    transformer = Transformer2DModel.from_pretrained(
        args.model_path,
        subfolder='transformer',
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )

    if args.merge_ratio > 0.0:
        model = patch.apply_patch(transformer,
                                  start_indices=args.start_indices,
                                  num_blocks=args.num_blocks,
                                  ratio=args.merge_ratio,
                                  sx=2, sy=2, latent_size=128, # change later

                                  unmerge_residual=args.unmerge_residual,
                                  cache_step=args.cache_step,
                                  push_unmerged=args.push_unmerged,

                                  hybrid_unmerge=args.hybrid_unmerge)

    pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        transformer=transformer,
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )
    pipe.to(device)

    prompts = PROMPT
    outputs = []
    total_runtime = 0.
    os.makedirs(args.experiment_folder, exist_ok=True)

    # multi-sampling
    for prompt in prompts:
        start = time.time()
        image = pipe(prompt,
                     height=1024,  # change later
                     width=1024,
                     num_inference_steps=args.sample_steps,
                     guidance_scale=args.guidance_scale
                     ).images[0]
        runtime = (time.time() - start)
        outputs.append(image)
        total_runtime += runtime
        patch.reset_cache(pipe)
        seed_everything(args.seed)

    # Save and display images:
    if args.merge_ratio > 0.0:
        if args.hybrid_unmerge > 0.0:
            save_path = (
                f"{args.experiment_folder}/hybrid_unmerge-{args.merge_ratio}-{args.start_indices}-threshold-{args.hybrid_unmerge}-"
                f"push_unmerged-{args.push_unmerged}.png")
        else:
            if args.unmerge_residual:
                save_path = (f"{args.experiment_folder}/cache_unmerge-{args.merge_ratio}-{args.start_indices}-"
                             f"push_unmerged-{args.push_unmerged}.png")
            else:
                save_path = f"{args.experiment_folder}/token_unmerge-{args.merge_ratio}-{args.start_indices}.png"
    else:
        save_path = f"{args.experiment_folder}/no-merge.png"

    save_grid(outputs, num_rows=1, output_path=save_path)
    print(f"Finish sampling {len(prompts)} images in {total_runtime} seconds.")
    print("Enjoy!")
