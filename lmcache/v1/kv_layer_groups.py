# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
import lmcache.c_ops as lmc_ops

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.utils import DiscoverableKVCache

logger = init_logger(__name__)


# The 4-tuple that uniquely identifies a set of kernel-equivalent layers:
# ``(kv_size, num_heads, head_size, block_size, dtype)``. Two layers share a transfer-
# kernel launch iff they share this identity — see the grouping loop in
# :meth:`KVLayerGroupsManager.__init__` for the derivation.
LayerGroupIdentity = tuple[int, int, int, int, torch.dtype]


@dataclass
class KVLayerGroupInfo:
    """A single transfer-kernel dispatch unit: a set of KV layers that can
    ride one kernel launch with one ``PageBufferShapeDesc``.

    Membership is decided by :class:`KVLayerGroupsManager` according to
    :data:`LayerGroupIdentity`; every layer referenced by
    ``layer_indices`` shares the same
    ``(kv_size, num_heads, head_size, block_size, dtype)`` signature.
    Consumers use ``layer_indices`` to pull the matching device pointers
    out of ``kv_caches`` (via
    :func:`~lmcache.v1.gpu_connector.utils.get_group_data_ptrs`) and
    feed them to the kernel alongside ``shape_desc``.

    ``dtype`` is carried alongside ``shape_desc`` because
    ``PageBufferShapeDesc.element_size`` is a byte width, which cannot
    distinguish dtypes that share a byte count (e.g. bfloat16 and
    float16 are both 2 bytes). Kernel template instantiation keys on the
    torch dtype, not the byte width, so we keep it explicit.

    Treat instances as immutable after construction; callers may hold
    references for the lifetime of the manager.
    """

    layer_indices: list[int]
    """0-based layer indices belonging to this group, in the order the
    kernel should iterate them. Fed to ``get_group_data_ptrs`` to build
    the per-group pointer array."""
    shape_desc: "lmc_ops.PageBufferShapeDesc"
    """Kernel-facing shape descriptor shared by every layer in the group.
    All seven fields (``kv_size, nl, nb, bs, nh, hs, element_size``) are
    stamped once at construction."""
    dtype: torch.dtype
    """Torch dtype of the KV cache tensors for this group. Used for
    kernel template instantiation; see class docstring for why we keep
    this alongside ``shape_desc.element_size``."""

    def __repr__(self) -> str:
        if not self.layer_indices:
            indices_repr = "[]"
        else:
            indices_repr = f"{self.layer_indices[0]}-{self.layer_indices[-1]}"
        sd = self.shape_desc
        return (
            f"KVLayerGroupInfo(layers={len(self.layer_indices)}, "
            f"indices={indices_repr}, "
            f"shape_desc=(kv={sd.kv_size}, nl={sd.nl}, nb={sd.nb}, "
            f"bs={sd.bs}, nh={sd.nh}, hs={sd.hs}, "
            f"element_size={sd.element_size}), dtype={self.dtype})"
        )

    @property
    def num_layers(self) -> int:
        """Number of layers in this group."""
        return len(self.layer_indices)

    @property
    def hidden_dim_size(self) -> int:
        """Hidden dimension size (``num_heads * head_size``)."""
        return self.shape_desc.nh * self.shape_desc.hs


class KVLayerGroupsManager:
    """Partition a model's KV layers into transfer-kernel dispatch units.

    At construction time, every layer in ``kv_caches`` is bucketed by its
    :data:`LayerGroupIdentity` (``(kv_size, num_heads, head_size,
    dtype)``). Each bucket becomes one :class:`KVLayerGroupInfo` holding
    the layer indices, a shared :class:`PageBufferShapeDesc`, and the
    group's torch dtype.

    Downstream consumers (``VLLMPagedMemGPUConnectorV3``,
    ``GPUCacheContext``, the multiprocess server) iterate
    ``self.kv_layer_groups`` and issue one transfer-kernel launch per
    group. The manager itself is a pure metadata object — it does not
    own any GPU buffers or perform any transfers.

    Layout parsing is delegated entirely to
    :mod:`lmcache.v1.gpu_connector.utils`; this class only drives the
    grouping and look-up.
    """

    # Formats whose per-layer tensor dim-0 is the *block* axis. These are
    # the only formats where ``tensor.stride(0)`` reflects the per-block
    # physical step that downstream kernels must honour to avoid dim-0
    # padding (e.g. mixed-compression KV pools). For every other format,
    # dim-0 is either KV, a token row (NBBS), or a layer-packed super-
    # block, and stride(0) has nothing to do with block-row padding —
    # propagating it would only confuse the kernel.
    _BLOCK_AXIS_FORMATS: frozenset = frozenset(
        {
            lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
            lmc_ops.GPUKVFormat.NL_X_NB_BS_HS,
        }
    )

    def __init__(
        self,
        kv_caches: "DiscoverableKVCache",
        gpu_kv_format: "lmc_ops.GPUKVFormat",
        num_blocks: int,
    ) -> None:
        """Partition layers into groups keyed by
        :data:`LayerGroupIdentity`.

        For each layer ``i`` in ``kv_caches``, read
        ``(kv_size, num_heads, head_size, dtype)`` via the format-aware
        accessors in ``utils.py``. Layers with identical identities are
        bucketed together; each bucket becomes one
        :class:`KVLayerGroupInfo`.

        Groups are emitted in the order of their first-appearing layer,
        so group indices are deterministic across runs.

        Args:
            kv_caches: KV cache structure accepted by
                :func:`normalize_kv_and_discover_format`.
            gpu_kv_format: Format returned by
                :func:`normalize_kv_and_discover_format`.
            num_blocks: Number of paged blocks. Stamped into every
                ``shape_desc.nb``. Each group's ``shape_desc.bs`` is
                discovered per-layer via :func:`get_block_size`, so
                compressed and non-compressed groups can coexist.
        """
        # Import here to break a circular import via
        # lmcache.v1.gpu_connector.__init__ → metadata → kv_layer_groups.
        # First Party
        from lmcache.v1.gpu_connector.utils import (
            get_num_layers,
            make_page_buffer_shape_desc,
        )

        self.kv_layer_groups: list[KVLayerGroupInfo] = []

        num_layers = get_num_layers(kv_caches, gpu_kv_format)
        if num_layers == 0:
            logger.debug("No KV caches available, skipping KV layer groups building")
            return

        groups_dict = self._group_layers_by_identity(
            kv_caches, gpu_kv_format, num_layers
        )

        # Emit groups in order of their first-appearing layer so that group
        # indices remain deterministic across runs.
        for (_, _, _, bs, dt), indices in sorted(
            groups_dict.items(), key=lambda kv: kv[1][0]
        ):
            rep = self._resolve_representative_tensor(
                kv_caches, gpu_kv_format, indices[0]
            )
            block_stride_elems = self._resolve_block_stride(
                rep, gpu_kv_format, indices[0]
            )
            shape_desc = make_page_buffer_shape_desc(
                kv_caches,
                gpu_kv_format,
                layer_idx=indices[0],
                num_layers_in_group=len(indices),
                num_blocks=num_blocks,
                block_size=bs,
                block_stride_elems=block_stride_elems,
            )
            self._log_representative_tensor_layout(rep, indices[0])
            self.kv_layer_groups.append(
                KVLayerGroupInfo(
                    layer_indices=indices,
                    shape_desc=shape_desc,
                    dtype=dt,
                )
            )

        logger.info("KV layer groups: %s", self.kv_layer_groups)

    @staticmethod
    def _group_layers_by_identity(
        kv_caches: "DiscoverableKVCache",
        gpu_kv_format: "lmc_ops.GPUKVFormat",
        num_layers: int,
    ) -> dict[LayerGroupIdentity, list[int]]:
        """Partition layer indices by :data:`LayerGroupIdentity`.

        Linear single pass over ``kv_caches``; layers sharing the same
        ``(kv_size, num_heads, head_size, block_size, dtype)`` signature
        land in the same bucket. The returned dict's value lists are
        later passed by reference into :class:`KVLayerGroupInfo`
        instances, so the dict itself is garbage-collected after
        ``__init__`` returns while the lists stay alive on each group.
        """
        # First Party
        from lmcache.v1.gpu_connector.utils import (
            get_block_size,
            get_dtype,
            get_head_size,
            get_num_heads,
            is_mla,
        )

        mla = is_mla(gpu_kv_format)
        kv_size = 1 if mla else 2
        groups_dict: dict[LayerGroupIdentity, list[int]] = defaultdict(list)
        for idx in range(num_layers):
            nh = 1 if mla else get_num_heads(kv_caches, gpu_kv_format, idx)
            hs = get_head_size(kv_caches, gpu_kv_format, idx)
            dt = get_dtype(kv_caches, gpu_kv_format, idx)
            bs = get_block_size(kv_caches, gpu_kv_format, idx)
            groups_dict[(kv_size, nh, hs, bs, dt)].append(idx)
        return groups_dict

    @staticmethod
    def _resolve_representative_tensor(
        kv_caches: "DiscoverableKVCache",
        gpu_kv_format: "lmc_ops.GPUKVFormat",
        layer_idx: int,
    ) -> torch.Tensor:
        """Return the ``torch.Tensor`` that physically holds the KV data
        for layer *layer_idx*, honouring the structural differences
        between formats.

        Callers use the returned tensor only to inspect layout
        (``shape`` / ``stride`` / ``storage_offset`` / ``dtype``) — it
        is *not* the K or V slice the kernel ingests directly.

        Format dispatch:

        * Cross-layer (``NB_NL_TWO_BS_NH_HS`` / ``NB_NL_TWO_NH_BS_HS``):
          ``kv_caches`` is itself the single backing tensor packing all
          layers along dim-1; indexing dim-0 (= NB) would yield a
          per-block slice, not a per-layer view, so we return the whole
          tensor. Its ``stride(0)`` is the authoritative per-NB step
          and is what ``_resolve_block_stride``'s non-block-axis padding
          check needs.
        * SGL MHA (``TWO_X_NL_X_NBBS_NH_HS``): outer list is K/V
          (length 2), inner list is per-layer — reach into
          ``kv_caches[0][layer_idx]`` for the layer's K tensor (K & V
          share shape/stride by construction).
        * Every other format: ``kv_caches`` is already a per-layer
          list; ``kv_caches[layer_idx]`` is the correct leaf.
        """
        if gpu_kv_format in (
            lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS,
            lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS,
        ):
            return kv_caches
        if gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
            return kv_caches[0][layer_idx]
        return kv_caches[layer_idx]

    @classmethod
    def _resolve_block_stride(
        cls,
        rep: torch.Tensor,
        gpu_kv_format: "lmc_ops.GPUKVFormat",
        layer_idx: int,
    ) -> "int | None":
        """Compute the per-block physical stride (in element units) to
        stamp into :class:`PageBufferShapeDesc` — or ``None`` if the
        format's dim-0 is not the block axis.

        For formats in :attr:`_BLOCK_AXIS_FORMATS`, ``tensor.stride(0)``
        is the authoritative per-block step; a value strictly larger
        than the tight stride means the group's row is dim-0-padded
        (e.g. DeepSeek V4 compressor / indexer caches sharing a KV pool
        with larger attn groups), and the downstream kernel will use it
        to step past padding bytes correctly.

        For all other formats, dim-0 is KV / NBBS / layer-packed and has
        no block-row semantics; ``None`` is returned so ``shape_desc``
        falls back to the tight stride. Meanwhile, if dim-0 IS padded in
        such a format, the downstream kernel would silently step with
        the tight stride and read/write garbage — so we fail loudly
        here with the representative tensor's full layout rather than
        corrupt KV bytes at transfer time.

        :class:`attempt_permute_to_contiguous_view` accepts any dim-0-
        only-padded view regardless of format because it has no format
        context; this method is that format-aware second line of
        defence.
        """
        if gpu_kv_format in cls._BLOCK_AXIS_FORMATS and rep.ndim > 0:
            return int(rep.stride(0))

        # Non-block-axis format: detect forbidden dim-0 padding.
        if rep.ndim >= 2:
            tight_dim0 = 1
            for d in range(1, rep.ndim):
                tight_dim0 *= int(rep.shape[d])
            padding = int(rep.stride(0)) - tight_dim0
            if padding > 0:
                raise ValueError(
                    "KVLayerGroupsManager: group's representative tensor "
                    f"has dim-0 padding ({padding} elements per block) but "
                    f"gpu_kv_format={gpu_kv_format!r} does not treat dim-0 "
                    "as the block axis (only NL_X_NB_TWO_BS_NH_HS and "
                    "NL_X_NB_BS_HS do); downstream transfer kernels cannot "
                    "honour this padding and would read/write wrong bytes. "
                    f"layer_idx={layer_idx}, shape={tuple(rep.shape)}, "
                    f"stride={tuple(rep.stride())}, "
                    f"tight_stride0={tight_dim0}, "
                    f"storage_offset={int(rep.storage_offset())}, "
                    f"dtype={rep.dtype}."
                )
        return None

    @staticmethod
    def _log_representative_tensor_layout(rep: torch.Tensor, layer_idx: int) -> None:
        """Dump shape / stride / contiguity / padding of the group's
        representative layer.

        For mixed-compression KV layouts (e.g. DeepSeek V4 compressor /
        indexer caches) the tensor can be a non-contiguous view over a
        larger storage; the stride pattern plus ``is_contiguous()``
        makes it obvious *which* dim is the offender, so downstream
        kernels (multi_layer_kv_transfer, attempt_permute_to_contiguous_view)
        can be reasoned about without re-deriving layout from raw shape
        alone.

        ``padding_per_block`` = ``stride[0] - prod(shape[1:])``; 0 means
        the block dim is tightly packed, non-zero exposes the per-block
        structural padding introduced by vLLM's KV allocator.
        """
        shape = tuple(rep.shape)
        stride = tuple(rep.stride())
        # Best-effort computations; the log line itself must not raise.
        try:
            inner = 1
            for s in shape[1:]:
                inner *= int(s)
            padding_per_block = stride[0] - inner if stride else 0
        except Exception:
            padding_per_block = -1
        try:
            storage_nbytes = rep.untyped_storage().nbytes()
        except Exception:
            storage_nbytes = -1
        logger.info(
            "Group first-layer tensor: layer_idx=%d shape=%s "
            "stride=%s is_contiguous=%s dtype=%s device=%s "
            "storage_offset=%d numel=%d storage_nbytes=%d "
            "padding_per_block=%d",
            layer_idx,
            shape,
            stride,
            rep.is_contiguous(),
            rep.dtype,
            rep.device,
            rep.storage_offset(),
            rep.numel(),
            storage_nbytes,
            padding_per_block,
        )

    @property
    def num_groups(self) -> int:
        """Number of :class:`KVLayerGroupInfo` entries.

        Zero if ``kv_caches`` had no layers at construction time.
        """
        return len(self.kv_layer_groups)

    def get_shape_desc(self, group_idx: int) -> "lmc_ops.PageBufferShapeDesc":
        """Return the :class:`PageBufferShapeDesc` for *group_idx*.

        Equivalent to ``self.kv_layer_groups[group_idx].shape_desc``.

        Args:
            group_idx: 0-based group index.

        Raises:
            IndexError: If *group_idx* is out of range.
        """
        return self.kv_layer_groups[group_idx].shape_desc
