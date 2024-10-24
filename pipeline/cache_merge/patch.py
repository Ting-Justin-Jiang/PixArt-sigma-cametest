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
    def __init__(self):
        self.rand_indices = {}  # key: index, value: rand_idx
        self.merge_path = {}  # key: index, value: step


class Cache:
    def __init__(self, index: int, cache_bus: CacheBus, broadcast_range: int = 1):
        self.cache_bus = cache_bus
        self.feature_map = None
        self.feature_map_broadcast = None
        self.feature_map_input = None
        self.index = index
        self.rand_indices = None
        self.broadcast_range = broadcast_range
        self.step = 0

    # == 1. Cache Merge Operations == #
    def push(self, x: torch.Tensor, index: torch.Tensor = None) -> None:
        # x would be the dst (updated) tokens during subsequent cache updates
        self.feature_map.scatter_(dim=-2, index=index, src=x.clone())
        logging.debug(f"\033[96mCache Push\033[0m: Push x: \033[95m{x.shape}\033[0m to cache index: \033[91m{self.index}\033[0m")

    def push_all(self, x: torch.Tensor) -> torch.Tensor:
        self.feature_map = x.clone()
        logging.debug(
            f"\033[96mCache Push\033[0m: Push All x: \033[95m{x.shape}\033[0m to cache index: \033[91m{self.index}\033[0m")
        return x

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
    # if reach this function, should perform merging at current step
    merge_config = cache.cache_bus.merge_path[cache.index]
    if cache.step in merge_config and (tome_info['args']['merge_start'] <= cache.step <= tome_info['args']['merge_end']):
        args = tome_info["args"]
        h, w = tome_info["size"]
        r = int(x.shape[1] * args["ratio"])

        # re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])

        # retrieve semi-random merging schedule
        if mode == 'cache_merge':
            if cache.rand_indices is None:
                cache.rand_indices = cache.cache_bus.rand_indices[cache.index][:]
            rand_indices = cache.rand_indices
        else:
            rand_indices = None

        m, u = merge.bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], r, tome_info,
                                                      no_rand=False, generator=args["generator"],
                                                      unmerge_mode=mode, cache=cache, rand_indices=rand_indices)
        return m, u

    elif (cache.step + 1 in merge_config or
          (cache.step + 2 in merge_config and tome_info["args"]["broadcast"])): # +2 because we may experience a broadcast
        return merge.do_nothing, cache.push_all

    else:
        return merge.do_nothing, merge.do_nothing


# = k based = #
def patch_attention(block_class: Type[torch.nn.Module], mode: str = 'token_merge') -> Type[torch.nn.Module]:
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
            m_a, u_a = compute_merge(k, self._mode, self._tome_info, self._cache)

            prune = self._tome_info['args']['prune']
            q, k, v = m_a(q, prune=prune), m_a(k, prune=prune), m_a(v, prune=prune)
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

            x = self.proj(x)
            x = self.proj_drop(x)

            # Unmerge. Cache at index is updated
            x = u_a(x)

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
            m_a, u_a = compute_merge(k_c, self._mode, self._tome_info, self._cache)
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
            if (self._tome_info['args']['broadcast'] and
                self._tome_info['args']['broadcast_start'] <= self._cache.step <= self._tome_info['args']['broadcast_end']
            ):
                if self._cache.should_save(self._tome_info['args']['broadcast_start']):
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
            if self._tome_info['args']['merge_cond']:
                return self.forward_merge_cond(x, mask, HW, block_id)

            if (self._tome_info['args']['broadcast'] and
                self._tome_info['args']['broadcast_start'] <= self._cache.step <= self._tome_info['args']['broadcast_end'] and
                self._cache.index in self._tome_info['args']['broadcast_blocks']
            ):
                if self._cache.should_save(self._tome_info['args']['broadcast_start']):
                    x = self.forward_merge_both(x, mask, HW, block_id)
                    self._cache.save(x)
                else:
                    x = self._cache.broadcast()
                    self._cache.step += 1
                return x

            return self.forward_merge_both(x, mask, HW, block_id)

    return PatchedAttentionKVCompress


# = x based = #
def patch_transformer(block_class: Type[torch.nn.Module], mode: str = 'token_merge') -> Type[torch.nn.Module]:
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
                m_a, u_a = compute_merge(x_a, self._mode, self._tome_info, self._cache)
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


def patch_diffuser_transformer(block_class: Type[torch.nn.Module], mode: str = 'token_merge') -> Type[torch.nn.Module]:
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


def yield_blocks(
    model: torch.nn.Module,
    start_indices: list[int],
    num_blocks: list[int],
    merge_metric: str
):
    def is_target_block(module):
        return isinstance_str(module, "PixArtMSBlock") or isinstance_str(module, "BasicTransformerBlock")

    current_block_index = 0
    target_indices = set()

    # Generate a set of all target block indices based on start_indices and num_blocks
    for start, num in zip(start_indices, num_blocks):
        block_indices = range(start, start + num)
        target_indices.update(block_indices)

    # Iterate over all modules in the model
    for name, module in model.named_modules():
        if is_target_block(module):
            current_block_index += 1
            if current_block_index in target_indices:
                if merge_metric == 'x':
                    # Yield the module itself
                    yield module, current_block_index
                elif merge_metric == 'k':
                    # Yield the attention sub-module ('attn' or 'attn1')
                    attn_module = getattr(module, 'attn', None) or getattr(module, 'attn1', None)
                    yield attn_module, current_block_index
                else:
                    raise ValueError(f"Invalid merge_metric: {merge_metric}")


def generate_semi_random_indices(sy: int, sx: int, h: int, w: int, steps: int) -> list:
    """
    Generates a semi-random merging schedule given the grid size.
    """
    hsy, wsx = h // sy, w // sx
    cycle_length = sy * sx
    num_cycles = -(-steps // cycle_length)

    num_positions = hsy * wsx

    # Generate random permutations for all positions
    random_numbers = torch.rand(num_positions, num_cycles * cycle_length)
    indices = random_numbers.argsort(dim=1)
    indices = indices[:, :steps] % cycle_length  # Map indices to [0, cycle_length - 1]

    # Reshape to (hsy, wsx, steps)
    indices = indices.view(hsy, wsx, steps)
    rand_idx = [indices[:, :, step].unsqueeze(-1) for step in range(steps)]

    return rand_idx


def reset_cache(pipeline: torch.nn.Module):
    is_diffusers = isinstance_str(pipeline, "PixArtSigmaPipeline")
    if is_diffusers:
        logging.info("Resetting model: Huggingface Diffuser")
        model = pipeline.transformer
    else:
        logging.info("Resetting model: source code")
        model = pipeline

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
        metric: str, ratio: float,
        mode: str, prune: bool,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        latent_size: int = 32,
        merge_cond: bool = False, # untested feature

        merge_step: Tuple[int, int] = (1, 19),
        cache_step: Tuple[int, int] = (1, 19),
        merge_path: Dict = Dict,
        push_unmerged: bool = True,

        # ==== 2. Broadcast ==== #
        broadcast_start_blocks: list[int] = list[int],
        broadcast_num_blocks: list[int] = list[int],
        broadcast_range: int = 1,
        broadcast_step: tuple[int, int] = (2, 18),
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
    assert len(broadcast_start_blocks) == len(broadcast_num_blocks), "Invalid broadcasting blocks"
    cache_start = max(cache_step[0], 1)  # Make sure the first step is token merging to avoid cache access

    if metric == 'x':
        merge_cond = False

    if broadcast_range <= 1:
        broadcast = False
        broadcast_blocks = []
    else:
        broadcast = True
        broadcast_blocks = []
        for start, num in zip(broadcast_start_blocks, broadcast_num_blocks):
            broadcast_blocks.extend(list(range(start, start + num)))

    is_diffusers = isinstance_str(model, "PixArtTransformer2DModel") # diffuser pipeline is somewhat broken right now
    if is_diffusers:
        logging.info("Patching PixArtMS from Huggingface Diffuser")
    elif isinstance_str(model, "PixArtMS"):
        logging.info("Patching PixArtMS from source code")
    else:
        raise RuntimeError("Provided model was not a PixArtMS model")

    logging.info(
        "\033[96mArguments:\033[0m\n"
        "\033[95m# ==== 1. Merging ==== #\033[0m\n"
        f"metric: {metric}\n"
        f"mode: {mode}\n"
        f"ratio: {ratio}\n"
        f"prune: {prune}\n"
        f"sx: {sx}, sy: {sy}\n"
        f"use_rand: {use_rand}\n"
        f"latent_size: {latent_size}\n"
        f"merge_cond: {merge_cond}\n"  # untested feature
        
        f"merge_path: {merge_path}\n"
        f"merge_step: {merge_step}\n"
        f"cache_step: {cache_start, cache_step[-1]}\n"
        f"push_unmerged: {push_unmerged}\n"
        
        "\033[95m# ==== 2. Broadcast ==== #\033[0m\n"
        f"broadcast: {broadcast}\n"
        f"broadcast_blocks: {broadcast_blocks}\n"
        f"broadcast_range: {broadcast_range}\n"
        f"broadcast_step: {broadcast_step[0], broadcast_step[1]}\n"
    )

    # == initialization ==
    remove_patch(model)
    model._tome_info = {
        "size": (latent_size // 2, latent_size // 2),
        "hooks": [],
        "args": {
            # 1. Merging #
            "metric": metric,
            "ratio": ratio,
            "mode": mode,
            "prune": prune,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "latent_size": latent_size,
            "merge_cond": merge_cond,

            "merge_path": merge_path,
            "merge_start": merge_step[0],
            "merge_end": merge_step[-1],
            "cache_start": cache_start,
            "cache_end": cache_step[-1],
            "push_unmerged": push_unmerged,

            # 2. Broadcasting #
            "broadcast": broadcast,
            "broadcast_range": broadcast_range,
            "broadcast_start": broadcast_step[0],
            "broadcast_end": broadcast_step[1],
            "broadcast_blocks": broadcast_blocks
        }
    }

    model._bus = CacheBus()

    # patch merging blocks
    model._bus.merge_path = merge_path
    for module, index in yield_blocks(model, [1], [28], metric):
        if metric == 'x':
            patch_fn = patch_diffuser_transformer if is_diffusers else patch_transformer
        elif metric == 'k':
            patch_fn = patch_attention
        else:
            raise ValueError(f'"{metric}" is not a valid metric')

        module.__class__ = patch_fn(module.__class__, mode=mode)
        module._tome_info = model._tome_info
        module._cache = Cache(index=index, cache_bus=model._bus, broadcast_range=broadcast_range)
        rand_indices = generate_semi_random_indices(module._tome_info["args"]['sy'],
                                                    module._tome_info["args"]['sx'],
                                                    module._tome_info["size"][0], module._tome_info["size"][1],
                                                    steps=merge_step[1] - merge_step[0] + 1)
        model._bus.rand_indices[module._cache.index] = rand_indices
        logging.debug(f'Applied merging patch at Block {index}')

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

