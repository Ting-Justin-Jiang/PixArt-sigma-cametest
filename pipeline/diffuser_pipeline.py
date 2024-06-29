import time
import os
import argparse
import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline
from pytorch_lightning import seed_everything

from cache_merge import patch
from prompt import PROMPT
from utils import save_grid


def get_args():
    parser = argparse.ArgumentParser()
    # == Model configuration == #
    parser.add_argument('--model_path', default='PixArt-alpha/PixArt-Sigma-XL-2-2K-MS', type=str)
    parser.add_argument('--seed', default=42, type=int, help='Seed for the random generator')
    parser.add_argument('--sample_steps', default=20, type=int, help='Number of inference steps')
    parser.add_argument('--guidance_scale', default=7.0, type=float, help='Guidance scale')

    # ==== ==== ==== ==== ==== ==== ==== ==== ==== #
    # ==== Token Merging Configuration ==== #
    parser.add_argument('--experiment-folder', type=str, default='samples/experiment/diffuser')
    parser.add_argument("--merge-ratio", type=float, default=0.4, help="Ratio of tokens to merge")
    parser.add_argument("--start-indices", type=lambda s: [int(item) for item in s.split(',')], default=[9, 21])
    parser.add_argument("--num-blocks", type=lambda s: [int(item) for item in s.split(',')], default=[8, 2])

    # == Improvements == #
    parser.add_argument("--semi-rand-schedule", action=argparse.BooleanOptionalAction, type=bool, default=False)
    parser.add_argument("--unmerge-residual", action=argparse.BooleanOptionalAction, type=bool, default=False)
    parser.add_argument("--push-unmerged", action=argparse.BooleanOptionalAction, type=bool, default=False)

    # == Hybrid Unmerge == #
    parser.add_argument("--hybrid-unmerge", type=float, default=0.0,
                        help="cosine similarity threshold, set 0.0 to bypass")

    # == Branch Features == #
    parser.add_argument("--upscale-guiding", type=int, default=0, help="guiding disable at, set 0 to bypass")
    parser.add_argument("--proportional-attention", action=argparse.BooleanOptionalAction, type=bool, default=False)

    return parser.parse_args()


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
                                  sx=2, sy=2, latent_size=256, # change later

                                  semi_rand_schedule=args.semi_rand_schedule,
                                  unmerge_residual=args.unmerge_residual,
                                  push_unmerged=args.push_unmerged,

                                  hybrid_unmerge=args.hybrid_unmerge,

                                  # == Branch Feature == #
                                  upscale_guiding=args.upscale_guiding,
                                  proportional_attention=args.proportional_attention)

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
                     height=2048,  # change later
                     width=2048,
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

    save_grid(outputs, num_rows=4, output_path=save_path)
    print(f"Finish sampling {len(prompts)} images in {total_runtime} seconds.")
    print("Enjoy!")
