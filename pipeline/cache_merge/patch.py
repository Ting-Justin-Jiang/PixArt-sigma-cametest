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


class Cache:
    def __init__(self, index: int, cache_bus: CacheBus, broadcast_range: int = 1):
        self.cache_bus = cache_bus
        self.feature_map = None
        self.feature_map_broadcast = None
        self.index = index
        self.broadcast_range = broadcast_range
        self.step = 0

    # == 1. Cache Merge Operations == #
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

    # == 2. Broadcast Operations == #
    def save(self, x: torch.Tensor) -> None:
        self.feature_map_broadcast = x.clone()

    def broadcast(self) -> torch.Tensor:
        if self.feature_map_broadcast is None:
            raise RuntimeError
        else:
            return self.feature_map_broadcast

    def should_save(self, broadcast_start: int) -> bool:
        if (self.step - broadcast_start) % self.broadcast_range == 0:
            logging.debug(f"\033[96mBroadcast\033[0m: Save at step: {self.step} cache index: \033[91m{self.index}\033[0m")
            return True  # Save at this step
        else:
            logging.debug(f"\033[96mBroadcast\033[0m: Broadcast at step: {self.step} cache index: \033[91m{self.index}\033[0m")
            return False # Broadcast at this step


def compute_merge(x: torch.Tensor, mode: str, tome_info: Dict[str, Any], cache: Cache) -> Tuple[Callable, ...]:
    args = tome_info["args"]
    h, w = tome_info["size"]
    r = int(x.shape[1] * args["ratio"])

    # Re-init the generator if it hasn't already been initialized or device has changed.
    if args["generator"] is None:
        args["generator"] = init_generator(x.device)
    elif args["generator"].device != x.device:
        args["generator"] = init_generator(x.device, fallback=args["generator"])

    # retrieve (or create) semi-random merging schedule
    if mode == 'cache_merge':
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
                                                  no_rand=False, generator=args["generator"],
                                                  unmerge_mode=mode, cache=cache, rand_indices=rand_indices)

    m_a, u_a = (m, u)
    m_m, u_m = (m, u) if args["merge_mlp"] else (merge.do_nothing, merge.do_nothing)

    return m_a, m_m, u_a, u_m


########################################################################################################################
# = k based = #
########################################################################################################################
def make_tome_attention(block_class: Type[torch.nn.Module], mode: str = 'token_merge') -> Type[torch.nn.Module]:
    class PatchedAttentionKVCompress(block_class):
        _parent = block_class
        _mode = mode

        def forward_merge_both(self, x, mask=None, HW=None, block_id=None):
            B, N, C = x.shape

            qkv = self.qkv(x).reshape(B, N, 3, C)
            q, k, v = qkv.unbind(2)
            dtype = q.dtype
            q = self.q_norm(q)
            k = self.k_norm(k)

            # Compute merge
            m_a, _, u_a, _ = compute_merge(k, self._mode, self._tome_info, self._cache)
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
            return x

        def forward_merge_cond(self, x, mask=None, HW=None, block_id=None):
            B, N, C = x.shape
            assert B == 2, "Only support single image per batch in this version"

            qkv = self.qkv(x).reshape(B, N, 3, C)
            q, k, v = qkv.unbind(2)
            dtype = q.dtype
            q = self.q_norm(q)
            k = self.k_norm(k)

            # Split the input metrics with conditional and unconditional
            q_c, q_uc = torch.split(q, 1, dim=0)
            k_c, k_uc = torch.split(k, 1, dim=0)
            v_c, v_uc = torch.split(v, 1, dim=0)

            del x, q, k, v

            # == 1. Conditioned ==
            # Compute merge configuration with metric k
            m_a, _, u_a, _ = compute_merge(k_c, self._mode, self._tome_info, self._cache)
            q_c, k_c, v_c = m_a(q_c), m_a(k_c), m_a(v_c)
            new_N = q_c.shape[1]

            # Reshape
            q_c = q_c.reshape(B // 2, new_N, self.num_heads, C // self.num_heads).to(dtype)
            k_c = k_c.reshape(B // 2, new_N, self.num_heads, C // self.num_heads).to(dtype)
            v_c = v_c.reshape(B // 2, new_N, self.num_heads, C // self.num_heads).to(dtype)

            q_uc = q_uc.reshape(B // 2, N, self.num_heads, C // self.num_heads).to(dtype)
            k_uc = k_uc.reshape(B // 2, N, self.num_heads, C // self.num_heads).to(dtype)
            v_uc = v_uc.reshape(B // 2, N, self.num_heads, C // self.num_heads).to(dtype)

            attn_bias = None
            if mask is not None:
                attn_bias = torch.zeros([B * self.num_heads, q_c.shape[1], k_c.shape[1]], dtype=q_c.dtype, device=q_c.device)
                attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float('-inf'))

            # Attention with cond (merged)
            x_c = xformers.ops.memory_efficient_attention(q_c, k_c, v_c, p=self.attn_drop.p, attn_bias=attn_bias)
            x_c = x_c.view(B // 2, new_N, C)
            # Unmerge. Cache at index is updated
            x_c = u_a(x_c)

            # == 2. Unconditioned ==
            if self._tome_info['args']['broadcast'] and self._tome_info['args']['broadcast_start'] <= self._cache.step <= self._tome_info['args']['broadcast_end']:
                should_save = self._cache.should_save(self._tome_info['args']['broadcast_start'])
                if should_save:
                    # Attention with uncond (unmerged)
                    x_uc = xformers.ops.memory_efficient_attention(q_uc, k_uc, v_uc, p=self.attn_drop.p, attn_bias=attn_bias)
                    x_uc = x_uc.view(B // 2, N, C)
                    self._cache.save(x_uc)
                else:
                    x_uc = self._cache.broadcast()
            else:
                x_uc = xformers.ops.memory_efficient_attention(q_uc, k_uc, v_uc, p=self.attn_drop.p, attn_bias=attn_bias)
                x_uc = x_uc.view(B // 2, N, C)

            x = torch.stack([x_c, x_uc]).squeeze(dim=1)

            x = self.proj(x)
            x = self.proj_drop(x)

            self._cache.step += 1
            return x

        def forward(self, x, mask=None, HW=None, block_id=None):
            merge_cond = self._tome_info['args']['merge_cond']
            if merge_cond:
                return self.forward_merge_cond(x, mask, HW, block_id)

            else:
                if self._tome_info['args']['broadcast_start'] <= self._cache.step <= self._tome_info['args']['broadcast_end']:
                    should_save = self._cache.should_save(self._tome_info['args']['broadcast_start'])
                    if should_save:
                        x = self.forward_merge_both(x, mask, HW, block_id)
                        self._cache.save(x)
                        return x
                    else:
                        x = self._cache.broadcast()
                        self._cache.step += 1
                        return x
                else:
                    x = self.forward_merge_both(x, mask, HW, block_id)
                    return x


    return PatchedAttentionKVCompress


########################################################################################################################
# = x based = #
########################################################################################################################
def make_tome_block(block_class: Type[torch.nn.Module], mode: str = 'token_merge') -> Type[torch.nn.Module]:
    class PatchedPixArtMSBlock(block_class):
        _parent = block_class
        _mode = mode

        def forward(self, x, y, t, mask=None, HW=None, **kwargs) -> torch.Tensor:
            B, N, C = x.shape
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)

            # 1. Self Attention
            if self._mode == 'token_merge' or self._mode == 'cache_merge':
                # == Merge == #
                # compute merge function (before soft-matching) doesn't give large overhead
                x_a = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
                m_a, _, u_a, _ = compute_merge(x_a, self._mode, self._tome_info, self._cache)
                x = x + self.drop_path(gate_msa * u_a(self.attn(m_a(x_a), HW=HW)))

            elif self._mode == 'broadcast':
                # == Broadcast == #
                if self._tome_info['args']['broadcast_start'] <= self._cache.step <= self._tome_info['args']['broadcast_end']:
                    should_save = self._cache.should_save(self._tome_info['args']['broadcast_start'])
                    if should_save:
                        x_a = self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), HW=HW)
                        self._cache.save(x_a)
                        x = x + self.drop_path(gate_msa * x_a)
                    else:
                        x_a = self._cache.broadcast()
                        x = x + self.drop_path(gate_msa * x_a)
                else:
                    # regular operation
                    x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), HW=HW))

            else:
                raise RuntimeError('Invalid accelerating mode')

            # 2. Cross Attention
            x = x + self.cross_attn(x, y, mask)

            # 3. MLP
            x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

            self._cache.step += 1
            return x

    return PatchedPixArtMSBlock


def make_diffusers_tome_block(block_class: Type[torch.nn.Module], mode: str = 'token_merge') -> Type[torch.nn.Module]:
    class PatchedBasicTransformerBlock(block_class):
        _parent = block_class
        _mode = mode

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
                m_a, _, u_a, _ = compute_merge(norm_hidden_states, self._mode, self._tome_info, self._cache)
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
                    yield attn_module, index
                else:
                    raise ValueError


def generate_semi_random_indices(sy: int, sx: int, h: int, w: int, steps: int) -> list:
    """
    generates a semi-random merging schedule given the grid size
    """
    hsy, wsx = h // sy, w // sx
    cycle_length = sy * sx
    num_cycles = (steps + cycle_length - 1) // cycle_length

    full_sequence = []
    rand_idx = []

    for _ in range(hsy * wsx):
        sequence = torch.cat([
            torch.randperm(cycle_length)
            for _ in range(num_cycles)
        ])
        full_sequence.append(sequence[:steps])

    for step in range(steps):
        step_tensor = torch.stack([seq[step] for seq in full_sequence]).reshape(hsy, wsx)
        rand_idx.append(step_tensor.unsqueeze(-1))

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
            broadcast_range = module._cache.broadcast_range # if it is a merging block, broadcast range will not be access
            module._cache = Cache(index=index, cache_bus=model._bus, broadcast_range=broadcast_range)
            logging.debug(f"\033[96mCache Reset\033[0m: for index: \033[91m{index}\033[0m")

    return pipeline


def apply_patch(
        model: torch.nn.Module,
        # ==== 1. Merging ==== #
        merge_metric: str,
        ratio: float,
        merge_cond: bool = False,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        merge_mlp: bool = False,
        latent_size: int = 32,

        # == 1.1 Token Merging (Spatial) == #
        start_indices: list[int] = list[int],
        num_blocks: list[int] = list[int],

        # == 1.2 Cache Merging (Spatial-Temporal) == #
        cache_start_indices: list[int] = list[int],
        cache_num_blocks: list[int] = list[int],
        cache_step: tuple[int] = (4, 15),
        push_unmerged: bool = False,

        # ==== 2. Broadcast (Temporal) ==== #
        broadcast_start_indices: list[int] = list[int],
        broadcast_num_blocks: list[int] = list[int],
        broadcast_range: int = 0,
        broadcast_step: tuple[int] = (2, 18),

        # == 3. Misc == #
        hybrid_unmerge: float = 0.0,
):

    # == merging preparation ==
    global DEBUG_MODE
    FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
    if DEBUG_MODE:
        torch.set_printoptions(threshold=100000)
        logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt='%Y-%m-%d %H:%M')
    else:
        logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%Y-%m-%d %H:%M')
    logging.debug('Start with \033[95mDEBUG\033[0m mode')
    logging.info('\033[94mApplying Token Merging\033[0m')

    # == assertions == #
    assert len(start_indices) == len(num_blocks), "Invalid merging blocks"
    assert len(cache_start_indices) == len(cache_num_blocks), "Invalid merging blocks"
    assert len(broadcast_start_indices) == len(broadcast_num_blocks), "Invalid merging blocks"

    if merge_metric == 'x':
        merge_cond = False

    if len(start_indices) != 0:
        token_merge = True
    else:
        token_merge = False

    if len(cache_start_indices) != 0:
        cache_merge = True
        cache_start = max(cache_step[0], 1) # Make sure the first step is token merging to avoid cache access
    else:
        cache_merge = False
        cache_start = max(cache_step[0], 1)
        # cache_start = None
        # cache_step = (None, None)
        # hybrid_unmerge = 0.0

    if broadcast_range < 1:
        broadcast = False
        broadcast_num_blocks = None
        broadcast_start_indices = None
        broadcast_step = (None, None)
    else:
        broadcast = True

    # Ensure provided model is PixArtMS
    is_diffusers = isinstance_str(model, "PixArtTransformer2DModel")
    if is_diffusers:
        logging.info("Patching model from Huggingface Diffuser")
    elif isinstance_str(model, "PixArtMS"):
        logging.info("Patching model from source code")
    else:
        raise RuntimeError("Provided model was not a PixArtMS model, as expected.")

    logging.info(
        "\033[96mArguments:\033[0m\n"
        "\033[95m# ==== 1. Merging ==== #\033[0m\n"
        f"merge_metric: {merge_metric}\n"
        f"merge_cond: {merge_cond}\n"
        f"ratio: {ratio}\n"
        f"sx: {sx}, sy: {sy}\n"
        f"use_rand: {use_rand}\n"
        f"merge_mlp: {merge_mlp}\n"
        f"latent_size: {latent_size}\n"
        
        "\033[95m# == 1.1 Token Merging (Spatial) == #\033[0m\n"
        f"token_merge:{token_merge}\n"
        f"start_indices: {start_indices}\n"
        f"num_blocks: {num_blocks}\n"
        
        "\033[95m# == 1.2 Cache Merging (Spatial-Temporal) == #\033[0m\n"
        f"cache_merge: {cache_merge}\n"
        f"cache_start_indices: {cache_start_indices}\n"
        f"cache_num_blocks: {cache_num_blocks}\n"
        f"push_unmerged: {push_unmerged}\n"
        f"cache_step: {cache_start, cache_step[-1]}\n"
        
        "\033[95m# ==== 2. Broadcast (Temporal) ==== #\033[0m\n"
        f"broadcast: {broadcast}\n"
        f"broadcast_start_indices: {broadcast_start_indices}\n"
        f"broadcast_num_blocks: {broadcast_num_blocks}\n"
        f"broadcast_range: {broadcast_range}\n"
        f"broadcast_step: {broadcast_step[0], broadcast_step[1]}\n"
        
        "\033[95m# == 3. Misc == #\033[0m\n"
        f"hybrid_unmerge: {hybrid_unmerge}\n"
    )

    # == initialization ==
    # Make sure the module is not currently patched
    remove_patch(model)

    model._tome_info = {
        "size": (latent_size // 2, latent_size // 2),
        "hooks": [],
        "args": {
            # ==== 1. Merging ==== #
            "merge_metric": merge_metric,
            "merge_cond": merge_cond,
            "ratio": ratio,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_mlp": merge_mlp,
            "latent_size": latent_size,

            # == 1.1 Token Merging (Spatial) == #
            "start_indices": start_indices,
            "num_blocks": num_blocks,

            # == 1.2 Cache Merging (Spatial-Temporal) == #
            "cache_merge" : cache_merge,
            "cache_start_indices": cache_start_indices,
            "cache_num_blocks": cache_num_blocks,
            "push_unmerged": push_unmerged,
            "cache_start": cache_start,
            "cache_end": cache_step[-1],

            # ==== 2. Broadcast (Temporal) ==== #
            "broadcast": broadcast,
            "broadcast_range": broadcast_range,
            "broadcast_start": broadcast_step[0],
            "broadcast_end": broadcast_step[1],
            "broadcast_start_indices": broadcast_start_indices,
            "broadcast_num_blocks": broadcast_num_blocks,

            # == Hybrid Unmerge == #
            "hybrid_unmerge": True if hybrid_unmerge > 0.0 else False,
            "hybrid_threshold": hybrid_unmerge if hybrid_unmerge > 0.0 else 0.0,
        }
    }

    model._bus = CacheBus()

    # == Patch PixArtMS Blocks==
    # == 1. Patch Token Merging Blocks ==
    if token_merge:
        for module, index in patch_tome_blocks(model, start_indices, num_blocks, merge_metric):
            if merge_metric == 'x':
                make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
            elif merge_metric == 'k':
                make_tome_block_fn = make_tome_attention
            else:
                raise ValueError(f'"{merge_metric}" is not a legal metric')

            module.__class__ = make_tome_block_fn(module.__class__, mode='token_merge')
            module._tome_info = model._tome_info

            module._cache = Cache(index=index, cache_bus=model._bus, broadcast_range=broadcast_range)
            logging.debug(f'Applied Token Merging patch at Block {index}')

    # == 2. Patch Cache Merging Blocks ==
    if cache_merge:
        for module, index in patch_tome_blocks(model, cache_start_indices, cache_num_blocks, merge_metric):
            if merge_metric == 'x':
                make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
            elif merge_metric == 'k':
                make_tome_block_fn = make_tome_attention
            else:
                raise ValueError(f'"{merge_metric}" is not a legal metric')

            module.__class__ = make_tome_block_fn(module.__class__, mode='cache_merge')
            module._tome_info = model._tome_info

            module._cache = Cache(index=index, cache_bus=model._bus, broadcast_range=broadcast_range)
            logging.debug(f'Applied Cache Merging patch at Block {index}')

    # == 3. Patch Broadcast Blocks ==
    if broadcast:
        for module, index in patch_tome_blocks(model, broadcast_start_indices, broadcast_num_blocks, 'x'):
            make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
            module.__class__ = make_tome_block_fn(module.__class__, mode='broadcast')
            module._tome_info = model._tome_info

            module._cache = Cache(index=index, cache_bus=model._bus, broadcast_range=broadcast_range)
            logging.debug(f'Applied Broadcast patch at Block {index}')

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

