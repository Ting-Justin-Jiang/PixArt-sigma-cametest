"""
Code adapted from original tomesd: https://github.com/dbolya/tomesd
Improved Token Merging for Diffusion Transformer
"""
DEBUG_MODE: bool = True
import torch
import logging
from . import merge
from .utils import isinstance_str, init_generator
from typing import Type, Dict, Any, Tuple, Callable
from diffusion.model.nets.PixArt_blocks import t2i_modulate


class CacheBus:
    """A Bus class for overall control."""
    def __init__(self):
        self.rand_indices = {}  # key: index, value: rand_idx

class Cache:
    def __init__(self, index: int, cache_bus: CacheBus):
        self.cache_bus = cache_bus
        self.feature_map = None
        self.index = index
        self.guiding_ratio = None
        self.step = 0

    def push(self, x: torch.Tensor, index: torch.Tensor = None) -> None:
        if self.feature_map is None:
            # x would be the entire feature map during the first cache update
            self.feature_map = x.clone()
            logging.debug(f"\033[96mCache Push\033[0m: Initial push x: \033[95m{x.shape}\033[0m to cache index: \033[91m{self.index}\033[0m")
        else:
            # x would be the dst (updated) tokens during subsequent cache updates
            self.feature_map.scatter_(dim=-2, index=index, src=x.clone())
            logging.debug(f"\033[96mCache Push\033[0m: Push x: \033[95m{x.shape}\033[0m to cache index: \033[91m{self.index}\033[0m")

    def pop(self, index: torch.Tensor) -> torch.Tensor:
        # Retrieve the src tokens from the cached feature map
        x = torch.gather(self.feature_map, dim=-2, index=index)
        logging.debug(f"\033[96mCache Pop\033[0m: Pop x: \033[95m{x.shape}\033[0m to cache index: \033[91m{self.index}\033[0m")
        return x


def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any], cache: Cache) -> Tuple[Callable, ...]:
    args = tome_info["args"]
    # x is ([2 * bs, 256, 1152]) if 256 model
    # x is ([2 * bs, 1024, 1152]) if 512 model
    # x is ([2 * bs, 16384, 1152]) if 2048 model
    h, w = tome_info["size"]
    if cache.guiding_ratio: # for upscale guidance
        r = int(x.shape[1] * cache.guiding_ratio)
    elif cache.guiding_ratio == 0.0:
        r = 0
    else:
        r = int(x.shape[1] * args["ratio"])

    # Re-init the generator if it hasn't already been initialized or device has changed.
    if args["generator"] is None:
        args["generator"] = init_generator(x.device)
    elif args["generator"].device != x.device:
        args["generator"] = init_generator(x.device, fallback=args["generator"])

    # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
    # batch, which causes artifacts with use_rand, so force it to be off.
    use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]

    # retrieve (or create) semi-random merging schedule
    if tome_info['args']['semi_rand_schedule']:
        if cache.index not in cache.cache_bus.rand_indices:
            rand_indices = generate_semi_random_indices(tome_info["args"]['sx'], tome_info["args"]['sy'], h, w, steps=250)
            cache.cache_bus.rand_indices[cache.index] = rand_indices
            logging.debug(
                f"\033[96mSemi Random Schedule\033[0m: Initial push to cache index: \033[91m{cache.index}\033[0m")
        else:
            rand_indices = cache.cache_bus.rand_indices[cache.index]
    else:
        rand_indices = None

    m, u = merge.bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], r, tome_info,
                                                  no_rand=not use_rand, generator=args["generator"], cache=cache, rand_indices=rand_indices)

    m_a, u_a = (m, u)
    m_m, u_m = (m, u) if args["merge_mlp"] else (merge.do_nothing, merge.do_nothing)

    return m_a, m_m, u_a, u_m


def make_guiding_attention(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class PatchedGuidingAttention(block_class):
        def forward(
                self, x: torch.Tensor, size: torch.Tensor = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
            B, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale

            # Apply proportional attention
            if size is not None:
                attn = attn + size.log()[:, None, None, :, 0]

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            return x

    return PatchedGuidingAttention



def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class PatchedPixArtMSBlock(block_class):
        _parent = block_class

        def forward(self, x, y, t, mask=None, HW=None, **kwargs) -> torch.Tensor:
            B, N, C = x.shape

            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)

            # compute merge function (before soft-matching) doesn't give large overhead
            x_a = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
            m_a, _, u_a, _ = compute_merge(x_a, self._tome_info, self._cache)
            x = x + self.drop_path(gate_msa * u_a(self.attn(m_a(x_a), HW=HW)))

            # cross attention and mlp
            x = x + self.cross_attn(x, y, mask)
            x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

            self._cache.step += 1
            return x

    return PatchedPixArtMSBlock


def make_guiding_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class PatchedGuidingBlock(block_class):
        _parent = block_class

        def forward(self, x: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
            # todo: Guiding Blocks need to be adapted

            # # Calculate the guidance ratio
            # step = self._cache.step
            # if step <= self._tome_info['args']['upscale_disable_after']:
            #     initial_ratio = 0.75
            #     self._cache.guiding_ratio = initial_ratio * (1 - (step - 1) / (self._tome_info['args']['upscale_disable_after'] - 1))
            #     if step >= 1:
            #         logging.debug(f"\033[93mUpscale Guiding ratio: {self._cache.guiding_ratio}\033[0m")
            # else:
            #     self._cache.guiding_ratio = 0.0
            #
            # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            #
            # x_a = modulate(self.norm1(x), shift_msa, scale_msa)
            # m_a, _, u_a, _ = compute_merge(x_a, self._tome_info, self._cache)
            #
            # m_x_a, size = merge.merge_wavg(m_a, x_a, self._tome_info['args']['proportional_attention']) # return merged x, size
            # x = x + gate_msa.unsqueeze(1) * u_a(self.attn(m_x_a, size))
            #
            # x_m = modulate(self.norm2(x), shift_mlp, scale_mlp)
            # x = x + gate_mlp.unsqueeze(1) * self.mlp(x_m)
            #
            # self._cache.step += 1
            return x

    return PatchedGuidingBlock


def patch_tome_blocks(model: torch.nn.Module, start_indices: list[int], num_blocks: list[int], upscale_guiding=False):
    index = 0
    block_ranges = []

    # Create ranges for each start_block and num_block
    for start, num in zip(start_indices, num_blocks):
        block_ranges.extend(range(start, start + num))

    for name, module in model.named_modules():
        if isinstance_str(module, "PixArtMSBlock"):
            index += 1
            if not upscale_guiding:
                if index in block_ranges:
                    yield module, index
            else:
                if index not in block_ranges:
                    attn = module.attn
                    yield module, attn, index


def generate_semi_random_indices(sy: int, sx: int, h: int, w: int, steps: int) -> torch.Tensor:
    """
    generates a semi-random merging schedule given the grid size
    """
    hsy, wsx = h // sy, w // sx
    cycle_length = sy * sx
    num_cycles = (steps + cycle_length - 1) // cycle_length

    full_sequence = []

    for _ in range(hsy * wsx):
        sequence = torch.cat([
            torch.randperm(cycle_length)
            for _ in range(num_cycles)
        ])
        full_sequence.append(sequence[:steps])

    full_sequence = torch.stack(full_sequence).to(torch.int64)
    rand_idx = full_sequence.reshape(hsy, wsx, steps).permute(2, 0, 1).unsqueeze(-1)
    return rand_idx


def reset_cache(model: torch.nn.Module):
    model._bus = CacheBus()
    for _, module in model.named_modules():
        if isinstance_str(module, "PatchedPixArtMSBlock") or isinstance_str(module, "PatchedGuidingBlock"):
            index = module._cache.index
            module._cache = Cache(index=index, cache_bus=model._bus)
            logging.debug(f"\033[96mCache Reset\033[0m: for index: \033[91m{index}\033[0m")
    return model


def apply_patch(
        model: torch.nn.Module,

        # == DiT blocks to merge == #
        start_indices: list[int],
        num_blocks: list[int],

        ratio: float = 0.5,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        merge_mlp: bool = False,
        latent_size: int = 32,

        # == Cache Merge ablation arguments == #
        semi_rand_schedule: bool = False,
        unmerge_residual: bool = False,
        push_unmerged: bool = False,

        # == Hybrid Merging == #
        hybrid_unmerge: float = 0.0,

        # == Branch Feature == #
        upscale_guiding: int = 0,
        proportional_attention: bool = True
):
    # == merging preparation ==
    # todo: Fix the latent_size with hooking for non-regular HW size
    global DEBUG_MODE
    FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
    if DEBUG_MODE:
        torch.set_printoptions(threshold=100000)
        logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt='%Y-%m-%d %H:%M')
    else:
        logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%Y-%m-%d %H:%M')
    logging.debug('Start with \033[95mDEBUG\033[0m mode')
    logging.info('\033[94mApplying Token Merging\033[0m')

    if hybrid_unmerge > 0.0:
        unmerge_residual = True # just an enforcement
        semi_rand_schedule = True

    if not upscale_guiding:
        proportional_attention = False

    logging.info(
        "\033[96mArguments:\n"
        f"start_indices: {start_indices}\n"
        f"num_blocks: {num_blocks}\n"
        f"ratio: {ratio}\n"
        f"sx: {sx}, sy: {sy}\n"
        f"use_rand: {use_rand}\n"
        f"merge_mlp: {merge_mlp}\n"
        f"latent_size: {latent_size}\n"
        f"semi_rand_schedule: {semi_rand_schedule}\n"
        f"unmerge_residual: {unmerge_residual}\n"
        f"push_unmerged: {push_unmerged}\n"
        f"hybrid_unmerge: {hybrid_unmerge > 0.0}\n"
        f"upscale_guiding: {upscale_guiding > 0}\n"
        f"proportional_attention: {proportional_attention}"
        f"\033[0m"
    )

    assert len(start_indices) == len(num_blocks), "Invalid merging argument"

    # == initialization ==
    # Make sure the module is not currently patched
    remove_patch(model)

    model._tome_info = {
        "size": (latent_size // 2, latent_size // 2),
        "hooks": [],
        "args": {
            "start_indices": start_indices,
            "num_blocks": num_blocks,
            "ratio": ratio,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_mlp": merge_mlp,
            "latent_size": latent_size,

            # == Cache Merge ablation arguments == #
            "semi_rand_schedule" : semi_rand_schedule,
            "unmerge_residual" : unmerge_residual,
            "push_unmerged": push_unmerged,

            # == Hybrid Unmerge == #
            "hybrid_unmerge": True if hybrid_unmerge > 0.0 else False,
            "hybrid_threshold": hybrid_unmerge if hybrid_unmerge > 0.0 else 0.0,

            # == Upscale Guiding == #
            "upscale_guiding": True if upscale_guiding > 0 else False,
            "upscale_disable_after": upscale_guiding if upscale_guiding > 0 else 0,
            "proportional_attention": proportional_attention
        }
    }

    model._bus = CacheBus()

    # Patch PixArtMS Blocks
    for module, index in patch_tome_blocks(model, start_indices, num_blocks):
        module.__class__ = make_tome_block(module.__class__)
        module._tome_info = model._tome_info

        module._cache = Cache(index=index, cache_bus=model._bus)
        logging.debug('Applied token merging patch at PixArtMSBlock %d', index)

    # Patch Guiding Blocks
    if model._tome_info['args']['upscale_guiding']:
        for module, attn, index in patch_tome_blocks(model, start_indices, num_blocks, upscale_guiding=True):
            module.__class__ = make_guiding_block(module.__class__)
            attn.__class__ = make_guiding_attention(attn.__class__)
            module._tome_info = model._tome_info

            module._cache = Cache(index=index, cache_bus=model._bus)
            logging.debug('Applied upscale guidance patch at DiTBlock %d', index)

    return model


def remove_patch(model: torch.nn.Module):
    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent

    return model

