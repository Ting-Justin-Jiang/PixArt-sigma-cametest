import torch
import math
from typing import Tuple, Callable
import logging


def do_nothing(x: torch.Tensor, mode: str = None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     tome_info: dict,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None,
                                     unmerge_mode: str = 'token_merge',
                                     cache: any = None,
                                     rand_indices: torch.Tensor = None) -> Tuple[Callable, Callable]:
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    # # todo: this is bad, consider replace in next update
    # if cache.step < tome_info['args']['cache_start'] or cache.step > tome_info['args']['cache_end']:
    #     if cache.step == tome_info['args']['cache_start'] - 1 and unmerge_mode == 'cache_merge':
    #         def initial_push(x: torch.Tensor):
    #             cache.push(x)
    #             return x
    #         return do_nothing, initial_push
    #     else:
    #         return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            if unmerge_mode == 'cache_merge':
                # retrieve from a pre-defined semi-random schedule
                rand_idx = rand_indices[cache.step].to(generator.device)
            else:
                rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)

        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx # this assumes we only choose one dst from a grid
        a_idx = rand_idx[:, num_dst:, :]  # src
        b_idx = rand_idx[:, :num_dst, :]  # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        spatial_scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = spatial_scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)  # dst indices src tokens should merge to

        if tome_info['args']['hybrid_unmerge']:
            threshold = tome_info['args']['hybrid_threshold']
            mask = node_max.gather(dim=-1, index=src_idx.squeeze(-1)) >= threshold
            r_c = math.floor(mask.sum(dim=-1).float().mean().item()) # this looks so bad ...

            # further partition of src_idx, dst_idx
            tome_src_idx, tome_dst_idx = (src_idx[..., :r_c, :], dst_idx[..., :r_c, :])
            came_src_idx, came_dst_idx = (src_idx[..., r_c:, :], dst_idx[..., r_c:, :])

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce_(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        # Simply concat
        out = torch.cat([unm, dst], dim=1)
        logging.debug(f"\033[96mMerge\033[0m: feature map merged from \033[95m{x.shape}\033[0m to \033[95m{out.shape}\033[0m "
                      f"at block index: \033[91m{cache.index}\033[0m")
        return out


    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        if unmerge_mode == 'cache_merge' and tome_info['args']['cache_start'] <= cache.step <= tome_info['args']['cache_end']:
            # == Branch 1: Improved Merging middle steps
            # Only proceed improved unmerging mechanism during middle steps

            if tome_info['args']['hybrid_unmerge']:
                # == SubBranch 1: Hybrid Unmerge
                # use hybrid unmerging: both cache and dup computation
                cache.push(dst, index=b_idx.expand(B, num_dst, c))
                if tome_info['args']['push_unmerged']:
                    cache.push(unm, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c))

                # Partition src_idx and dst_idx
                came_src_idx_expanded = gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=came_src_idx).expand(B, r - r_c, c)
                came_src = cache.pop(index=came_src_idx_expanded)

                tome_dst_idx_expanded = tome_dst_idx.expand(B, r_c, c)
                tome_src = gather(dst, dim=-2, index=tome_dst_idx_expanded)

                logging.debug(f"\033[92mHybrid Unmerging: duplicate approximate (tome) src token {tome_src.shape}, cached approximate (came) src token {came_src.shape}\033[0m")

                # == Combine back to the original shape (Branch 1 SubBranch 1)
                out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
                out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
                out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
                out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=tome_src_idx).expand(B, r_c, c), src=tome_src)
                out.scatter_(dim=-2, index=came_src_idx_expanded, src=came_src)
                return out

            else:
                # == SubBranch 2: Cache Unmerge
                cache.push(dst, index=b_idx.expand(B, num_dst, c))
                if tome_info['args']['push_unmerged']:
                    cache.push(unm, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c))
                src = cache.pop(index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c))

        else:
            # == Branch 2: Token Merging & Improved Merging first/last steps
            # Proceed vanilla token unmerging, while updating the cache if cache unmerging is enabled
            if unmerge_mode == 'cache_merge' and cache.feature_map is not None:
                cache.push(dst, index=b_idx.expand(B, num_dst, c))
                if tome_info['args']['push_unmerged']:
                    cache.push(unm, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c))

            src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # == Combine back to the original shape (Branch 1 SubBranch 2 & Branch 2)
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        # For the first step
        if unmerge_mode == 'cache_merge' and cache.feature_map is None:
            cache.push(out)

        return out

    return merge, unmerge