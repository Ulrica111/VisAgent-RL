# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import inspect
import logging
import os
from contextlib import contextmanager
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from tensordict import TensorDict

_PATCHED_VLLM_GET_CACHED_TOKENIZER = False
_PATCHED_VLLM_TRANSFORMERS_IMPL_COMPAT = False
_PATCHED_VLLM_TRANSFORMERS_MODEL_QWEN3_VL = False
_PATCHED_HF_FLASH_ATTN_S_AUX_NONE = False
_PATCHED_VLLM_QWEN3_VL_INPUT_PREPROCESSOR = False
_PATCHED_VLLM_QWEN3_VL_MODEL_REGISTRY_MM = False
_PATCHED_VLLM_MM_TRANSFORMERS_DEFAULT_MAPPERS = False


def _patch_multimodal_registry_huggingface_mappers_for_transformers_model() -> None:
    """``ImagePlugin`` only dispatches if a mapper is registered per model class; HF ``TransformersModel`` has none, causing KeyError after Qwen3-VL is marked multimodal."""
    global _PATCHED_VLLM_MM_TRANSFORMERS_DEFAULT_MAPPERS
    if _PATCHED_VLLM_MM_TRANSFORMERS_DEFAULT_MAPPERS:
        return
    try:
        from vllm.model_executor.models.transformers import TransformersModel
        from vllm.multimodal import MULTIMODAL_REGISTRY
    except ImportError:
        return

    MULTIMODAL_REGISTRY.register_image_input_mapper()(TransformersModel)
    MULTIMODAL_REGISTRY.register_input_mapper("video", None)(TransformersModel)
    _PATCHED_VLLM_MM_TRANSFORMERS_DEFAULT_MAPPERS = True


def _patch_vllm_model_registry_qwen3_vl_mark_multimodal() -> None:
    """Mark Qwen3-VL as multimodal in vLLM so ``ModelConfig`` gets ``MultiModalConfig`` and ``init_mm_limits_per_prompt`` uses non-zero modality limits.

    HF ``TransformersModel`` resolves with ``supports_multimodal=False``; without this, multimodal payloads still reach ``MULTIMODAL_REGISTRY.map_input`` but limits stay at 0 (`--limit-mm-per-prompt`-style disabled map), yielding "image=0 ... but found 1 items".

    Rollout pins ``VLLM_USE_V1=0`` for Qwen3-VL to avoid V1 multimodal-engine edge cases tied to Transformers+V1.
    """
    global _PATCHED_VLLM_QWEN3_VL_MODEL_REGISTRY_MM
    if _PATCHED_VLLM_QWEN3_VL_MODEL_REGISTRY_MM:
        return
    try:
        import vllm.model_executor.models.registry as vllm_model_registry
    except ImportError:
        return
    _orig = vllm_model_registry._ModelRegistry.is_multimodal_model

    def is_multimodal_model(self, architectures):  # type: ignore[no-untyped-def,misc]
        arch_list = architectures if isinstance(architectures, list) else list(architectures or ())
        for a in arch_list:
            if isinstance(a, str) and "Qwen3VL" in a:
                return True
        return _orig(self, architectures)

    vllm_model_registry._ModelRegistry.is_multimodal_model = is_multimodal_model  # type: ignore[method-assign,misc]
    _PATCHED_VLLM_QWEN3_VL_MODEL_REGISTRY_MM = True


def _patch_vllm_input_preprocessor_qwen3_vl_legacy_mm() -> None:
    """Prefer legacy ``token_inputs(..., multi_modal_data=...)`` for Qwen3-VL Transformers (no vLLM multimodal processor).

    Together with `_patch_vllm_model_registry_qwen3_vl_mark_multimodal`: ``model_config.is_multimodal_model`` is true while ``MULTIMODAL_REGISTRY.has_processor`` stays false — `_orig` returns False and keeps legacy dict ingestion.
    """
    global _PATCHED_VLLM_QWEN3_VL_INPUT_PREPROCESSOR
    if _PATCHED_VLLM_QWEN3_VL_INPUT_PREPROCESSOR:
        return
    try:
        from vllm.inputs.preprocess import InputPreprocessor
    except ImportError:
        return

    _orig = InputPreprocessor._can_process_multimodal

    def _is_qwen3_vl_model_config(model_config: Any) -> bool:
        hf = getattr(model_config, "hf_config", None)
        if hf is None:
            return False
        if getattr(hf, "model_type", None) == "qwen3_vl":
            return True
        archs = getattr(hf, "architectures", None) or []
        return any(isinstance(a, str) and "Qwen3VL" in a for a in archs)

    def _can_process_multimodal(self) -> bool:  # type: ignore[no-untyped-def,misc]
        mc = self.model_config
        if not mc.is_multimodal_model and _is_qwen3_vl_model_config(mc):
            return False
        return _orig(self)

    InputPreprocessor._can_process_multimodal = _can_process_multimodal  # type: ignore[method-assign,misc]
    _PATCHED_VLLM_QWEN3_VL_INPUT_PREPROCESSOR = True


def _patch_vllm_get_cached_tokenizer_for_hf5() -> None:
    """vLLM get_cached_tokenizer expects all_special_tokens_extended; transformers 5.x TokenizersBackend (e.g. Qwen2Tokenizer) omits it."""
    global _PATCHED_VLLM_GET_CACHED_TOKENIZER
    if _PATCHED_VLLM_GET_CACHED_TOKENIZER:
        return
    try:
        from vllm.transformers_utils import tokenizer as vllm_tokenizer_mod
    except ImportError:
        return

    _orig = vllm_tokenizer_mod.get_cached_tokenizer

    def get_cached_tokenizer(tokenizer):  # type: ignore[misc,no-redef]
        if not hasattr(tokenizer, "all_special_tokens_extended"):
            ext = list(getattr(tokenizer, "all_special_tokens", []) or [])
            tokenizer.all_special_tokens_extended = ext  # type: ignore[attr-defined]
        return _orig(tokenizer)

    vllm_tokenizer_mod.get_cached_tokenizer = get_cached_tokenizer  # type: ignore[assignment]
    _PATCHED_VLLM_GET_CACHED_TOKENIZER = True


def _patch_vllm_transformers_impl_compat_for_hf_attention_backend() -> None:
    """When ``supports_backend`` is absent vLLM used ``_supports_flex_attn``, which falsely rejects HF models that only set ``_supports_attention_backend`` + ``is_backend_compatible`` (e.g. Qwen3-VL)."""
    global _PATCHED_VLLM_TRANSFORMERS_IMPL_COMPAT
    if _PATCHED_VLLM_TRANSFORMERS_IMPL_COMPAT:
        return
    try:
        import vllm.model_executor.model_loader.utils as vllm_model_loader_utils
    except ImportError:
        return
    _orig = vllm_model_loader_utils.is_transformers_impl_compatible

    def is_transformers_impl_compatible(arch, module=None):  # type: ignore[misc,no-redef]
        import transformers

        mod = module or getattr(transformers, arch, None)
        if mod is None:
            return False
        if getattr(mod, "_supports_attention_backend", False):
            try:
                return bool(mod.is_backend_compatible())
            except Exception:
                return False
        return _orig(arch, module)

    vllm_model_loader_utils.is_transformers_impl_compatible = is_transformers_impl_compatible  # type: ignore[assignment]
    _PATCHED_VLLM_TRANSFORMERS_IMPL_COMPAT = True


def _ensure_qwen3_vl_hf_config_lm_fields(hf_config: Any) -> None:
    """Qwen3-VL nests LM dimensions under ``text_config``; vLLM ``TransformersModel`` reads ``vocab_size`` / ``hidden_size`` / layer counts on the root config."""
    if getattr(hf_config, "model_type", None) != "qwen3_vl":
        return
    tc = getattr(hf_config, "text_config", None)
    if tc is None:
        return
    for key in (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rms_norm_eps",
        "hidden_act",
        "intermediate_size",
    ):
        if not hasattr(hf_config, key) and hasattr(tc, key):
            setattr(hf_config, key, getattr(tc, key))


def _pick_qwen3_vl_subconfig_for_module(class_name: str, hf_root: Any) -> Any:
    if class_name.startswith("Qwen3VLVision"):
        return hf_root.vision_config
    if class_name.startswith("Qwen3VLText"):
        return hf_root.text_config
    return hf_root


def _rebuild_hf_module_for_vllm_meta_buffer(hf_root: Any, module: torch.nn.Module) -> torch.nn.Module:
    """vLLM ``TransformersModel.init_buffers`` uses ``type(module)(hf_config)``, which breaks Qwen3-VL layers that take ``(dim, ...)``, ``LayerNorm``, or vision/text sub-configs instead of the root ``Qwen3VLConfig``."""
    if isinstance(module, torch.nn.LayerNorm):
        return type(module)(
            module.normalized_shape,
            eps=module.eps,
            elementwise_affine=module.elementwise_affine,
            bias=module.bias is not None,
        )

    cls = type(module)
    name = cls.__name__

    if name == "Qwen3VLVisionRotaryEmbedding":
        return cls(module.dim, getattr(module, "theta", 10000.0))
    if name == "Qwen3VLTextRMSNorm":
        return cls(int(module.weight.shape[0]), float(module.variance_epsilon))

    sig = inspect.signature(cls.__init__)
    param_list = [(k, v) for k, v in sig.parameters.items() if k != "self"]
    if not param_list:
        return cls()

    first_name, _ = param_list[0]
    if first_name != "config":
        return cls(hf_root)

    mt = getattr(hf_root, "model_type", None)
    cfg_call = _pick_qwen3_vl_subconfig_for_module(name, hf_root) if mt == "qwen3_vl" else hf_root

    extra_args: List[Any] = []
    for pname, param in param_list[1:]:
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            break
        val = getattr(module, pname, inspect.Parameter.empty)
        if val is inspect.Parameter.empty and param.default is not inspect.Parameter.empty:
            val = param.default
        elif val is inspect.Parameter.empty:
            val = None
        extra_args.append(val)

    return cls(cfg_call, *extra_args)


def _hf_flash_attn_implementation_for_kernel(module: torch.nn.Module) -> str:
    """``_flash_attention_forward`` / ``lazy_import_flash_attention`` only understand HF kernel names (e.g. ``flash_attention_2``); configs built under vLLM may still say ``vllm``."""
    impl = getattr(module.config, "_attn_implementation", None) or "flash_attention_2"
    suffix = impl.split("|")[-1]
    if suffix != "vllm":
        return impl
    try:
        from transformers.utils import is_flash_attn_2_available, is_flash_attn_3_available, is_flash_attn_4_available

        if is_flash_attn_4_available():
            return "flash_attention_4"
        if is_flash_attn_3_available():
            return "flash_attention_3"
        if is_flash_attn_2_available():
            return "flash_attention_2"
    except ImportError:
        pass
    return "flash_attention_2"


def _patch_hf_flash_attention_forward_allow_none_s_aux() -> None:
    """HF ``flash_attention_forward`` always does ``s_aux.to(dtype)``; ``s_aux`` defaults to None (e.g. Qwen3-VL vision)."""
    global _PATCHED_HF_FLASH_ATTN_S_AUX_NONE
    if _PATCHED_HF_FLASH_ATTN_S_AUX_NONE:
        return
    try:
        import transformers.integrations.flash_attention as hf_fa_mod
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
    except ImportError:
        return

    def flash_attention_forward(  # type: ignore[no-untyped-def,misc]
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling=None,
        sliding_window=None,
        softcap=None,
        is_causal=None,
        s_aux=None,
        **kwargs,
    ):
        if kwargs.get("output_attentions", False):
            hf_fa_mod.logger.warning_once(
                "Flash Attention does not support `output_attentions=True`."
                " Please set your attention to `eager` if you want any of these features."
            )

        seq_len = query.shape[2]

        if any(dim == 0 for dim in query.shape):
            raise ValueError(
                "Tensor query has shape  with a zero dimension.\n"
                "FlashAttention does not support inputs with dim=0.\n"
                "Please check your input shapes or use SDPA instead."
            )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        target_dtype = hf_fa_mod.get_target_dtype(query, module)

        is_causal = is_causal if is_causal is not None else module.is_causal

        s_aux_cast = s_aux.to(query.dtype) if s_aux is not None else None

        attn_output = _flash_attention_forward(
            query,
            key,
            value,
            attention_mask,
            query_length=seq_len,
            is_causal=is_causal,
            dropout=dropout,
            softmax_scale=scaling,
            sliding_window=sliding_window,
            softcap=softcap,
            use_top_left_mask=hf_fa_mod._use_top_left_mask,
            target_dtype=target_dtype,
            attn_implementation=_hf_flash_attn_implementation_for_kernel(module),
            layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
            s_aux=s_aux_cast,
            **kwargs,
        )

        return attn_output, None

    hf_fa_mod.flash_attention_forward = flash_attention_forward  # type: ignore[assignment,misc]
    _PATCHED_HF_FLASH_ATTN_S_AUX_NONE = True


def _patch_vllm_transformers_model_init_for_qwen3_vl() -> None:
    global _PATCHED_VLLM_TRANSFORMERS_MODEL_QWEN3_VL
    if _PATCHED_VLLM_TRANSFORMERS_MODEL_QWEN3_VL:
        return
    try:
        import vllm.model_executor.models.transformers as vllm_transformers_backend
    except ImportError:
        return
    _orig = vllm_transformers_backend.TransformersModel.__init__

    def TransformersModel__init__(self, *, vllm_config, prefix: str = "") -> None:  # type: ignore[no-untyped-def,misc]
        _ensure_qwen3_vl_hf_config_lm_fields(vllm_config.model_config.hf_config)
        _orig(self, vllm_config=vllm_config, prefix=prefix)

    def init_buffers(self, module: torch.nn.Module) -> None:  # type: ignore[no-untyped-def,misc]
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.device == torch.device("meta"):
                rebuilt = _rebuild_hf_module_for_vllm_meta_buffer(self.config, module)
                new_buffer = getattr(rebuilt, name)
                setattr(module, name, new_buffer)
        for child in module.children():
            self.init_buffers(child)

    vllm_transformers_backend.TransformersModel.__init__ = TransformersModel__init__  # type: ignore[method-assign,misc]
    vllm_transformers_backend.TransformersModel.init_buffers = init_buffers  # type: ignore[method-assign,misc]

    # ``TransformersModel`` builds HF weights with attn_implementation="vllm" for *all* blocks. vLLM's hook indexes
    # KV by ``module.layer_idx`` (decoder-only); Qwen3-VL vision attention has no ``layer_idx`` and uses varlen FA
    # kwargs (`cu_seq_lens_*`, …). Delegate those modules back to HF ``flash_attention_forward``.
    _orig_vllm_fa = vllm_transformers_backend.vllm_flash_attention_forward

    def vllm_flash_attention_forward_compat(  # type: ignore[no-untyped-def,misc]
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling=None,
        attention_instances=None,
        **kwargs,
    ):
        if not hasattr(module, "layer_idx"):
            kwargs.pop("attention_instances", None)
            from transformers.integrations.flash_attention import flash_attention_forward as hf_flash_attn

            return hf_flash_attn(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
                **kwargs,
            )

        return _orig_vllm_fa(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            attention_instances=attention_instances,
            **kwargs,
        )

    vllm_transformers_backend.vllm_flash_attention_forward = vllm_flash_attention_forward_compat  # type: ignore[method-assign,misc]
    import transformers.modeling_utils as _tfmu

    _tfmu.ALL_ATTENTION_FUNCTIONS["vllm"] = vllm_flash_attention_forward_compat  # type: ignore[index,misc]

    _HF_FORWARD_MM_KEYS = frozenset(
        (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
            "mm_token_type_ids",
            "attention_mask",
        )
    )

    def TransformersModel_forward(  # type: ignore[no-untyped-def,misc]
        self,
        input_ids: Union[torch.Tensor, None],
        positions: torch.Tensor,
        intermediate_tensors: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        """Relay HF multimodal tensors into the underlying transformers model (vLLM stub dropped ``**kwargs``)."""
        from vllm.distributed import get_pp_group
        from vllm.sequence import IntermediateTensors as _IntermediateTensors
        from vllm.transformers_utils.processor import cached_processor_from_config

        hf_mm = {k: v for k, v in kwargs.items() if k in _HF_FORWARD_MM_KEYS and v is not None}

        if not get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            input_ids = None
            inputs_embeds = intermediate_tensors["hidden_states"]

        ids_batched_for_mm: Optional[torch.Tensor] = None
        if input_ids is not None:
            ids_batched_for_mm = input_ids[None, ...]
            input_ids = ids_batched_for_mm

        if (
            getattr(self.config, "model_type", None) == "qwen3_vl"
            and ids_batched_for_mm is not None
            and hf_mm.get("mm_token_type_ids") is None
            and (
                hf_mm.get("image_grid_thw") is not None or hf_mm.get("video_grid_thw") is not None
            )
        ):
            try:
                processor = cached_processor_from_config(self.model_config)
                if hasattr(processor, "create_mm_token_type_ids"):
                    seq = ids_batched_for_mm[0].tolist()
                    mm_cpu = processor.create_mm_token_type_ids([seq])
                    hf_mm["mm_token_type_ids"] = torch.tensor(
                        mm_cpu, device=ids_batched_for_mm.device, dtype=torch.int32
                    )
            except Exception as exc:
                logger.warning("qwen3_vl multimodal: could not synthesize mm_token_type_ids (%s)", exc)

        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds[None, ...]

        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            position_ids=positions[None, ...],
            attention_instances=self.attention_instances,
            return_dict=False,
            **hf_mm,
        )[0][0, ...]

        if not get_pp_group().is_last_rank:
            return _IntermediateTensors({"hidden_states": hidden_states})

        return hidden_states

    vllm_transformers_backend.TransformersModel.forward = TransformersModel_forward  # type: ignore[method-assign,misc]

    _PATCHED_VLLM_TRANSFORMERS_MODEL_QWEN3_VL = True


_patch_vllm_model_registry_qwen3_vl_mark_multimodal()
_patch_multimodal_registry_huggingface_mappers_for_transformers_model()
_patch_vllm_input_preprocessor_qwen3_vl_legacy_mm()
_patch_vllm_get_cached_tokenizer_for_hf5()
_patch_vllm_transformers_impl_compat_for_hf_attention_backend()
_patch_hf_flash_attention_forward_allow_none_s_aux()
_patch_vllm_transformers_model_init_for_qwen3_vl()

from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.model import get_hf_max_position_embeddings
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from verl.workers.agent import agent_rollout_loop

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _vllm_registry_supports_limit_mm_per_prompt(model_hf_config) -> bool:
    """Only pass limit_mm_per_prompt when vLLM recognizes the arch as multimodal (see vllm ModelConfig._init_multimodal_config)."""
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        archs = getattr(model_hf_config, "architectures", None)
        if not archs:
            return False
        return bool(ModelRegistry.is_multimodal_model(archs))
    except Exception:
        return False


def _requires_vllm_v0_engine_for_hf_qwen3_vl_rollout(model_hf_config) -> bool:
    """Qwen3-VL under Transformers backend uses legacy ``token_inputs(multi_modal_data=...)`` dicts; vLLM V1 rejects them (expects ``MultiModalKwargs``)."""
    if getattr(model_hf_config, "model_type", None) == "qwen3_vl":
        return True
    archs = getattr(model_hf_config, "architectures", None) or []
    return any(isinstance(a, str) and "Qwen3VL" in a for a in archs)


def _ensure_vllm_v0_engine_for_hf_qwen3_vl_rollout(model_hf_config) -> None:
    if not _requires_vllm_v0_engine_for_hf_qwen3_vl_rollout(model_hf_config):
        return
    prev = os.environ.get("VLLM_USE_V1")
    if prev not in (None, "0"):
        logger.warning(
            "Overriding VLLM_USE_V1=%s to 0: Qwen3-VL multimodal rollout passes "
            "legacy multi_modal_data dicts; vLLM V1 Processor requires MultiModalKwargs.",
            prev,
        )
    os.environ["VLLM_USE_V1"] = "0"


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), (
            "disable CUDA graph (enforce_eager = False) if free cache engine"
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(
                    tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp
                )
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        hf_max_pos = get_hf_max_position_embeddings(model_hf_config)
        assert hf_max_pos >= config.prompt_length + config.response_length, (
            "model context length should be greater than total sequence length"
        )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len:
            logger.warning(
                "max_num_batched_tokens (%s) < max_model_len (%s); "
                "lifting max_num_batched_tokens to max_model_len (vLLM SchedulerConfig).",
                max_num_batched_tokens,
                max_model_len,
            )
            max_num_batched_tokens = max_model_len

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = {}
        if _vllm_registry_supports_limit_mm_per_prompt(model_hf_config):
            if config.get("limit_images", None):
                limit_mm_per_prompt["image"] = config.get("limit_images")
            if config.get("agent") is not None and config.agent.activate_agent and config.agent.max_vllm_images:
                limit_mm_per_prompt["image"] = config.agent.max_vllm_images
                limit_mm_per_prompt["video"] = config.agent.max_vllm_videos

        _ensure_vllm_v0_engine_for_hf_qwen3_vl_rollout(model_hf_config)

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            # limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            # max_model_len=max_model_len + 16384,
            # max_model_len=32768,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=self.config.get("enable_prefix_caching", True),
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **({"limit_mm_per_prompt": limit_mm_per_prompt} if limit_mm_per_prompt else {}),
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        self.pad_token_id = tokenizer.pad_token_id

        self.tokenizer = tokenizer

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                if 'image' in multi_modal_data and multi_modal_data['image']:
                    vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
                else:
                    vllm_inputs.append({"prompt_token_ids": raw_prompt_ids})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        max_turn_of_validation = prompts.meta_info.get("max_turn_of_validation", None)
        print(f"########################## data meta-info: {prompts.meta_info}")
        print(f"########################## is validate: {is_validate}")
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
                "max_turn_of_validation": max_turn_of_validation
            }
            # self.sampling_params.n = 1

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            if self.config.agent.activate_agent:
                agent_proto = agent_rollout_loop(
                    config=self.config,
                    vllm_engine=self.inference_engine,
                    vllm_inputs=vllm_inputs, 
                    prompts=prompts,
                    multi_modal_inputs=non_tensor_batch.get("multi_modal_inputs", None),
                    sampling_params=self.sampling_params,
                    max_turn_of_validation=max_turn_of_validation
                )
                response = agent_proto.batch.pop('response')
            else:
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )

                # TODO(sgm): disable logprob when recompute_log_prob is enable
                # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

                response = []
                for output in outputs:
                    for sample_id in range(len(output.outputs)):
                        response.append(output.outputs[sample_id].token_ids)

                response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                    idx.device
                )

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                        non_tensor_batch["multi_modal_inputs"], self.sampling_params.n
                    )

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        # if position_ids.dim() == 3:  # qwen2vl mrope
        #     delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope (batch size, 4, seq len)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        if 'raw_prompt' in non_tensor_batch.keys():
            non_tensor_batch.pop('raw_prompt')
        if 'multi_modal_data' in non_tensor_batch.keys():
            non_tensor_batch.pop('multi_modal_data')
        if 'origin_multi_modal_data' in non_tensor_batch.keys():
            non_tensor_batch.pop('origin_multi_modal_data', None)

        if self.config.agent.activate_agent:
            batch = batch.update(agent_proto.batch)
            non_tensor_batch.update(agent_proto.non_tensor_batch)
            tool_name_key = self.config.agent.tool_name_key
            if tool_name_key and tool_name_key in non_tensor_batch.keys():
                non_tensor_batch.pop(tool_name_key)
            print(f' [DEBUG agent output proto] {batch.keys()=}, {non_tensor_batch.keys()=}')

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
