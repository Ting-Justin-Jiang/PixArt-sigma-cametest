"""
Code adapted from original tomesd: https://github.com/dbolya/tomesd
Improved Token Merging for Diffusion Transformer
"""
DEBUG_MODE: bool = True
import torch
import logging
import xformers.ops

from . import merge
from .utils import isinstance_str, init_generator
from typing import Optional, Type, Dict, Any, Tuple, Callable
from diffusion.model.nets.PixArt_blocks import t2i_modulate


class CacheBus:
    """A Bus class for overall control."""
    def __init__(self):
        self.rand_indices = {}  # key: index, value: rand_idx

        self.proj_cached_metrics = {}
        self.temporal_scores = {}


class Cache:
    def __init__(self, index: int, cache_bus: CacheBus):
        self.cache_bus = cache_bus
        self.feature_map = None
        self.index = index
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
    h, w = tome_info["size"]
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
    if tome_info['args']['unmerge_residual']:
        if cache.index not in cache.cache_bus.rand_indices:
            rand_indices = generate_semi_random_indices(tome_info["args"]['sy'], tome_info["args"]['sx'], h, w, steps=20)
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


########################################################################################################################
# = k based = #
########################################################################################################################
def make_tome_attention(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class PatchedAttentionKVCompress(block_class):
        _parent = block_class

        def forward(self, x, mask=None, HW=None, block_id=None):
            B, N, C = x.shape

            qkv = self.qkv(x).reshape(B, N, 3, C)
            q, k, v = qkv.unbind(2)
            dtype = q.dtype
            q = self.q_norm(q)
            k = self.k_norm(k)

            if self._tome_info['args']['temporal_score'] and self._cache.index - 1 in self._cache.cache_bus.proj_cached_metrics:
                # todo: fix

                # Computes similarity of k(I+1)(t-1) with K(cached(I+1)(t-1)), where cached(I+1)(t-1) approximates x(I+1)(t)
                with torch.no_grad():
                    # Retrieve k(I+1)(t-1) ~ k(I+1)(t)
                    k_metric = k.clone()
                    k_metric = k_metric / k_metric.norm(dim=-1, keepdim=True)

                    # Retrieve K(cached(I+1)(t-1))
                    cached_qkv = self.qkv(self._cache.cache_bus.proj_cached_metrics[self._cache.index-1]).reshape(B, N, 3, C)
                    _, cached_k_metric, _ = cached_qkv.unbind(2)
                    cached_k_metric = cached_k_metric / cached_k_metric.norm(dim=-1, keepdim=True)

                    # Pipeline: free up memory
                    self._cache.cache_bus.proj_cached_metrics[self._cache.index - 1] = None

                    temporal_score = k_metric @ cached_k_metric.transpose(-1, -2)
                    temporal_score = temporal_score.diagonal(dim1=-2, dim2=-1)  # Get diagonal elements which correspond to the similarity of each index

                    logging.debug(f"Temporal similarity mean: {temporal_score.mean(dim=1)}")
                    logging.debug(f"Temporal similarity max: {temporal_score.max(dim=1)[0]}")

                    # Pipeline: save to I (last) cache
                    self._cache.cache_bus.temporal_scores[self._cache.index - 1] = temporal_score

                    del k_metric, cached_qkv, cached_k_metric


            # Compute merge
            m_a, _, u_a, _ = compute_merge(k, self._tome_info, self._cache)
            q, k, v = m_a(q), m_a(k), m_a(v)
            new_N = q.shape[1]

            q = q.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)
            k = k.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)
            v = v.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)

            attn_bias = None
            if mask is not None:
                attn_bias = torch.zeros([B * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
                attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float('-inf'))
            x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

            x = x.view(B, new_N, C)

            # Unmerge. Cache at index is updated
            x = u_a(x)

            x = self.proj(x)
            x = self.proj_drop(x)

            self._cache.step += 1

            if self._tome_info['args']['temporal_score'] and self._cache.step >= self._tome_info['args']['cache_start']:
                # todo: fix

                cached_metric = self._cache.feature_map.clone()
                cached_metric = self.proj_drop(self.proj(cached_metric))
                self._cache.cache_bus.proj_cached_metrics[self._cache.index] = cached_metric
                del cached_metric

            return x

    return PatchedAttentionKVCompress


########################################################################################################################
# = x based = #
########################################################################################################################
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


def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class PatchedBasicTransformerBlock(block_class):
        _parent = block_class

        def forward(self,
                    hidden_states: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    encoder_hidden_states: Optional[torch.Tensor] = None,
                    encoder_attention_mask: Optional[torch.Tensor] = None,
                    timestep: Optional[torch.LongTensor] = None,
                    cross_attention_kwargs: Dict[str, Any] = None,
                    class_labels: Optional[torch.LongTensor] = None,
                    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
                    ) -> torch.Tensor:
            """
            Adapted from Huggingface Diffuser
            """

            # 0. Self-Attention
            batch_size = hidden_states.shape[0]

            if self.norm_type == "ada_norm_single":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                        self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                m_a, _, u_a, _ = compute_merge(norm_hidden_states, self._tome_info, self._cache)
                norm_hidden_states = norm_hidden_states.squeeze(1)
            else:
                raise ValueError("Incorrect norm used.")

            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}

            attn_output = u_a(self.attn1(
                m_a(norm_hidden_states),
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            ))
            attn_output = gate_msa * attn_output

            hidden_states = attn_output + hidden_states


            # 1. Cross-Attention
            if self.attn2 is not None:
                if self.norm_type == "ada_norm_single":
                    # For PixArt norm2 isn't applied here:
                    # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                    norm_hidden_states = hidden_states
                else:
                    raise ValueError("Incorrect norm")

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states


            # 2. Feed-forward
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp * ff_output

            hidden_states = ff_output + hidden_states

            self._cache.step += 1

            return hidden_states


    return PatchedBasicTransformerBlock


def patch_tome_blocks(model: torch.nn.Module, start_indices: list[int], num_blocks: list[int], merge_metric: str):
    index = 0
    block_ranges = []

    # Create ranges for each start_block and num_block
    for start, num in zip(start_indices, num_blocks):
        block_ranges.extend(range(start, start + num))

    for name, module in model.named_modules():
        if isinstance_str(module, "PixArtMSBlock") or isinstance_str(module, "BasicTransformerBlock"):
            index += 1
            if index in block_ranges:
                if merge_metric == 'x':
                    yield module, index
                elif merge_metric == 'k':
                    attn_module = getattr(module, 'attn', None) or getattr(module, 'attn1', None)
                    cross_attn_module = getattr(module, 'cross_attn', None) or getattr(module, 'attn2', None)
                    yield attn_module, index
                else:
                    raise ValueError


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


def reset_cache(pipeline: torch.nn.Module):

    is_diffusers = isinstance_str(pipeline, "PixArtSigmaPipeline")
    if is_diffusers:
        logging.info("Resetting model: Huggingface Diffuser")
        model = pipeline.transformer
    else:
        logging.info("Resetting model: source code")
        model = pipeline

    model._bus = CacheBus()
    for _, module in model.named_modules():
        if (isinstance_str(module, "PatchedPixArtMSBlock")
            or isinstance_str(module, "PatchedBasicTransformerBlock")
            or isinstance_str(module, "PatchedAttentionKVCompress")
        ):
            index = module._cache.index
            module._cache = Cache(index=index, cache_bus=model._bus)
            logging.debug(f"\033[96mCache Reset\033[0m: for index: \033[91m{index}\033[0m")

    return pipeline


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
        unmerge_residual: bool = False,
        cache_step: tuple[int] = (4, 15),
        push_unmerged: bool = False,

        # == Hybrid Merging == #
        hybrid_unmerge: float = 0.0,

        # == Merge Metric == #
        merge_metric: str = 'x',

        # == Temporal Score == #
        temporal_score: bool = False
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

    cache_start = max(cache_step[0], 1) # another enforcement, make sure the first step is token merging to avoid cache access

    # Ensure provided model is PixArtMS
    is_diffusers = isinstance_str(model, "PixArtTransformer2DModel")
    if is_diffusers:
        logging.info("Patching model from Huggingface Diffuser")
    elif isinstance_str(model, "PixArtMS"):
        logging.info("Patching model from source code")
    else:
        raise RuntimeError("Provided model was not a PixArtMS model, as expected.")

    logging.info(
        "\033[96mArguments:\n"
        f"start_indices: {start_indices}\n"
        f"num_blocks: {num_blocks}\n"
        f"ratio: {ratio}\n"
        f"sx: {sx}, sy: {sy}\n"
        f"use_rand: {use_rand}\n"
        f"merge_mlp: {merge_mlp}\n"
        f"latent_size: {latent_size}\n"
        f"unmerge_residual: {unmerge_residual}\n"
        f"push_unmerged: {push_unmerged}\n"
        f"cache_steps: {cache_start, cache_step[-1]}\n"
        f"hybrid_unmerge: {hybrid_unmerge > 0.0}\n"
        f"merge_metric: {merge_metric}\n",
        f"temporal_score: {temporal_score}\n",
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
            "unmerge_residual" : unmerge_residual,
            "push_unmerged": push_unmerged,
            "cache_start": cache_start,
            "cache_end": cache_step[-1],

            # == Hybrid Unmerge == #
            "hybrid_unmerge": True if hybrid_unmerge > 0.0 else False,
            "hybrid_threshold": hybrid_unmerge if hybrid_unmerge > 0.0 else 0.0,

            # == Merge Metric == #
            "merge_metric": merge_metric,

            # == Temporal Score == #
            "temporal_score": temporal_score
        }
    }

    model._bus = CacheBus()

    # Patch PixArtMS Blocks
    for module, index in patch_tome_blocks(model, start_indices, num_blocks, merge_metric):
        if merge_metric == 'x':
            make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
        elif merge_metric == 'k':
            make_tome_block_fn = make_tome_attention
        else:
            raise ValueError(f'"{merge_metric}" is not a legal metric')

        module.__class__ = make_tome_block_fn(module.__class__)
        module._tome_info = model._tome_info

        module._cache = Cache(index=index, cache_bus=model._bus)
        logging.debug(f'Applied token merging patch at Block {index}')

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

