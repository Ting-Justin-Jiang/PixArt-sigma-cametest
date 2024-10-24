"""
Proceed global greedy algorithm and metropolis hastings to search the optimal path for merging
"""
import argparse
import copy
import random
import math
import json
import inference_pipeline

import matplotlib.pyplot as plt
import numpy as np

from prompt import prompts

# preparation
initial_path = {i: [] for i in range(1, 29)}


def get_args():
    parser = argparse.ArgumentParser(description="Parse arguments for the token merging algorithm.")
    parser.add_argument('--num-blocks', type=int, default=14)
    parser.add_argument('--num-steps', type=int, default=20)
    parser.add_argument('--start-step', type=int, default=0)
    parser.add_argument('--mh-iterations', type=int, default=84)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--output', type=str, default="paths.json")

    args = parser.parse_args()
    return args


def plot_merge_heatmap(path, num_steps, title='Merge Heatmap', save_path=None):
    blocks = sorted(path.keys())
    heatmap = np.zeros((len(blocks), num_steps))

    for i, block in enumerate(blocks):
        for step in path[block]:
            heatmap[i, step] = 1

    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap, aspect='auto', cmap='Greys', interpolation='none')
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel('Blocks')
    plt.colorbar(label='Merged (1) or Not Merged (0)')
    plt.yticks(ticks=range(len(blocks)), labels=blocks)

    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Figure saved as {save_path}")

    plt.show()


def find_path(
    prompt: str,
    num_blocks: int = 14,
    num_steps: int = 20,
    start_step: int = 0,
    mh_iterations: int = 84,
    temperature: float = 0.05
):
    """
    Combines Greedy Search with Metropolis-Hastings to find an optimal token merging path.

    Parameters:
        prompt (str): The input prompt for the diffusion model.
        num_blocks (int): Number of blocks to select in each greedy step.
        num_steps (int): Total number of steps in the search.
        start_step (int): Starting step index.
        mh_iterations (int): Number of MH iterations per step.
        temperature (float): Temperature parameter for MH acceptance probability.

    Returns:
        path (dict): Optimized path dictionary mapping blocks to merge steps.
        lpips_log (dict): Dictionary logging LPIPS scores during MH phases.
    """
    # Initialize the path
    path = copy.deepcopy(initial_path)
    list_blocks = list(range(1, 29))

    # Initialize LPIPS logging dictionary
    # Structure: {step: [lpips_iter1, lpips_iter2, ..., lpips_iterN]}
    lpips_log = {step: [] for step in range(start_step, num_steps)}

    # Generate Baseline
    inference_pipeline.main(prompts=[prompt], merge_ratio=0.0, merge_path=path)
    print("Generated Baseline")

    for step in range(start_step, num_steps):
        lpips_scores = {}

        # --- Greedy Selection Phase ---
        for block in list_blocks:
            temp_path = copy.deepcopy(path)
            temp_path.setdefault(block, []).append(step)  # Ensure block key exists
            current_lpips = inference_pipeline.main(
                prompts=[prompt],
                merge_ratio=0.5,
                merge_path=temp_path
            )
            lpips_scores[block] = current_lpips

        # Select the top num_blocks with the lowest LPIPS scores
        sorted_blocks = sorted(lpips_scores.items(), key=lambda x: x[1])
        selected_blocks = [block for block, _ in sorted_blocks[:num_blocks]]

        # Update the path with Greedy Selection
        for block in selected_blocks:
            path.setdefault(block, []).append(step)

        print(f"Step {step}, selected blocks (Greedy): {selected_blocks}")
        print(f"Path after Greedy: {path}")

        # --- Metropolis-Hastings (MH) Phase ---
        # Compute current_lpips once before MH iterations
        current_lpips = inference_pipeline.main(
            prompts=[prompt],
            merge_ratio=0.5,
            merge_path=path
        )
        best_path = path
        best_lpips = current_lpips

        for mh_iter in range(mh_iterations):
            # **Proposal Step: Swap Assignments Between Blocks**
            current_assigned_blocks = [
                block for block in list_blocks if step in path.get(block, [])
            ]
            current_unassigned_blocks = [
                block for block in list_blocks if step not in path.get(block, [])
            ]

            if not current_assigned_blocks or not current_unassigned_blocks:
                print("  MH Iteration skipped: No possible swap candidates.")
                lpips_log[step].append(current_lpips)
                continue

            block_to_remove = random.choice(current_assigned_blocks)
            block_to_add = random.choice(current_unassigned_blocks)

            proposal_path = copy.deepcopy(path)
            proposal_path[block_to_remove].remove(step)
            proposal_path.setdefault(block_to_add, []).append(step)

            # **Compute LPIPS for Proposal**
            proposal_lpips = inference_pipeline.main(
                prompts=[prompt],
                merge_ratio=0.5,
                merge_path=proposal_path
            )

            # **Calculate Delta**
            delta = proposal_lpips - current_lpips

            # **Calculate Acceptance Probability**
            acceptance_prob = min(1, math.exp(-delta / temperature))

            # **Accept or Reject Proposal**
            random_value = random.random()
            if random_value < acceptance_prob:
                path = proposal_path
                current_lpips = proposal_lpips
                print(f"  MH Iteration {mh_iter + 1}: Accepted swap ({block_to_remove} ↔ {block_to_add}) | ΔLPIPS = {delta:.4f} | α = {acceptance_prob:.4f}")
                if proposal_lpips < best_lpips:
                    best_path = proposal_path
                    best_lpips = proposal_lpips
                    print(f"  MH Iteration {mh_iter + 1}: Best LPIPS updated to {proposal_lpips:.4f}")
                lpips_log[step].append(current_lpips)

            else:
                lpips_log[step].append(current_lpips)
                print(f"  MH Iteration {mh_iter + 1}: Rejected swap ({block_to_remove} ↔ {block_to_add}) | ΔLPIPS = {delta:.4f} | α = {acceptance_prob:.4f}")


        # After MH iterations, set path to the best_path found during MH
        path = best_path
        print(f"Path after MH: {path}\n")

    return path, lpips_log


def main():
    args = get_args()
    paths = {}

    prompt = prompts[0]  # computation-sake
    print(f"Finding path for prompt: {prompt}")
    path, lpips_log = find_path(prompt,
                                args.num_blocks,
                                args.num_steps,
                                args.start_step,
                                args.mh_iterations,
                                args.temperature
                                )
    print(f"Finished finding Path: {path}")
    paths[prompt] = path

    # Save paths
    serializable_paths = {
        prompt: {str(block): steps for block, steps in path.items()}
        for prompt, path in paths.items()
    }
    with open(args.output, 'w') as f:
        json.dump(serializable_paths, f, indent=4)

    # Save logging
    with open(f"lpips_{args.num_blocks}.json", 'w') as json_file:
        json.dump(lpips_log, json_file, indent=4)

    # Load paths
    with open(args.output, 'r') as f:
        serializable_paths = json.load(f)
    paths = {
        prompt: {int(block): steps for block, steps in path.items()}
        for prompt, path in serializable_paths.items()
    }

    plot_merge_heatmap(paths[prompts[-1]], num_steps=20, title=f'Merge Heatmap for "{prompts[-1]}"', save_path=f"heatmap_{args.num_blocks}.png")


if __name__ == '__main__':
    main()