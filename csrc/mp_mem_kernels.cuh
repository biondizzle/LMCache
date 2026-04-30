// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mem_kernels.cuh"  // TransferDirection, GPUKVFormat

#include <c10/cuda/CUDAGuard.h>
#include <vector>

struct PageBufferShapeDesc {
  int kv_size;       // 1 or 2
  int nl;            // num layers
  int nb;            // num blocks
  int bs;            // block size
  int nh;            // num heads
  int hs;            // head size
  int element_size;  // bytes (1 or 2)
  // Physical per-block stride in element units (= tensor.stride(0) of the
  // representative layer). 0 means "unset — consumer must fall back to the
  // format-specific tight stride". For vLLM KV pools where a group's row is
  // padded up to the pool's max row width (e.g. DeepSeek V4 compressor /
  // indexer caches sharing storage with larger attn groups), this value is
  // strictly larger than the tight stride; the kernel MUST use it to avoid
  // stepping into padding bytes.
  //
  // Only meaningful for formats whose dim-0 is the block axis:
  //   - NL_X_NB_TWO_BS_NH_HS   (per-layer [NB, 2, BS, NH, HS])
  //   - NL_X_NB_BS_HS          (per-layer [NB, BS, HS], MLA)
  // For NB_NL_TWO_BS_NH_HS (single tensor, dim-0 packs all layers),
  // NL_X_TWO_NB_BS_NH_HS (dim-0 is KV), and the SGL formats (dim-0 is the
  // token row NBBS), per-block padding at dim-0 does not exist and the
  // field is ignored.
  int block_stride_elems;

  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_head() const {
    return hs * element_size / sizeof(ScalarType);
  }

  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_token() const {
    return nh * hs * element_size / sizeof(ScalarType);
  }

  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_block() const {
    return bs * nh * hs * element_size / sizeof(ScalarType);
  }

  // Per-block stride in ScalarType units, for formats whose dim-0 is the
  // block axis. ``tight`` is the format-specific fallback when
  // ``block_stride_elems`` is 0 (untouched by caller / legacy path).
  template <typename ScalarType>
  __host__ __device__ inline size_t block_stride_or(
      size_t tight_in_scalar_type) const {
    if (block_stride_elems <= 0) {
      return tight_in_scalar_type;
    }
    // block_stride_elems counts elements of the source KV dtype
    // (element_size bytes each); convert to ScalarType units.
    return static_cast<size_t>(block_stride_elems) * element_size /
           sizeof(ScalarType);
  }
};

template <typename ScalarType>
struct MemoryObj4 {
  ScalarType* objects[4];
  int num_objects;  // 0 - 4
};

/**
 * Block-level multi-layer KV transfer between vLLM paged buffers and
 * LMCache contiguous memory objects.
 *
 * @param paged_buffer_ptrs_tensor  GPU int64 tensor of data pointers into
 *                                  vLLM paged buffers (one per tensor)
 * @param lmcache_objects_ptrs      Raw pointers to LMCache memory objects
 * @param block_ids                 GPU int64 tensor of block indices in vLLM
 *                                  paged buffer
 * @param device                    CUDA device of vLLM tensors
 * @param direction                 H2D (LMCache->vLLM) or D2H (vLLM->LMCache)
 * @param shape_desc                Shape descriptor for the paged buffer
 * @param lmcache_chunk_size        Tokens per LMCache memory object
 * @param gpu_kv_format             GPUKVFormat identifier
 * @param skip_prefix_n_blocks      Number of blocks to skip at the beginning
 */
void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    GPUKVFormat gpu_kv_format, int skip_prefix_n_blocks);
