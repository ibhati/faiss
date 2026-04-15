/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Portions Copyright 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/svs/IndexSVSFaissUtils.h>
#include <faiss/svs/IndexSVSVamana.h>

#include <svs/runtime/api_defs.h>
#include <svs/runtime/vamana_index.h>

#include <string>

namespace faiss {

/// Data placement for individual storage components.
enum SVSDataPlacement {
    /// Data component resides in heap RAM (default).
    SVS_PLACEMENT_RAM = 0,
    /// Data component is memory-mapped from SSD (zero-copy, lazy page-in).
    SVS_PLACEMENT_SSD = 1,
};

/// Search parameters for IndexSVSVamanaSSD (static, read-only index).
struct SearchParametersSVSVamanaSSD : public SearchParameters {
    size_t search_window_size = 0;
    size_t search_buffer_capacity = 0;
};

/// A static, read-only SVS Vamana index assembled from a saved directory.
///
/// This index uses VamanaIndex::assemble_from_directory() to load a
/// previously-built and saved index. When configured with SSD placement,
/// the quantized data (LVQ/LeanVec) is memory-mapped from disk instead
/// of being copied into RAM, enabling zero-copy loading for large datasets.
///
/// Primary and secondary data placement can be controlled independently:
///   - primary_placement:   LeanVec reduced-dim / LVQ main quantized data
///   - secondary_placement: LeanVec full-dim / LVQ residual correction data
///
/// Unlike IndexSVSVamana (which wraps DynamicVamanaIndex and supports add/remove),
/// this index is immutable after loading.
///
/// Example usage:
///   auto* idx = new IndexSVSVamanaSSD(
///       128, METRIC_L2, SVS_LeanVec4x8,
///       "/path/to/saved_index",
///       "/mnt/nvme",
///       SVS_PLACEMENT_RAM,   // primary (reduced-dim) in RAM
///       SVS_PLACEMENT_SSD    // secondary (full-dim) on SSD
///   );
///   idx->search(nq, xq, k, distances, labels);
///
struct IndexSVSVamanaSSD : Index {
    SVSStorageKind storage_kind;

    /// Search parameters
    size_t search_window_size = 10;
    size_t search_buffer_capacity = 10;

    /// SSD configuration
    std::string saved_directory;
    std::string ssd_path;
    SVSDataPlacement primary_placement = SVS_PLACEMENT_RAM;
    SVSDataPlacement secondary_placement = SVS_PLACEMENT_RAM;

    /// Enable primary-only mode for LeanVec storage.
    /// When true, only the reduced-dimension primary data is used
    /// (no reranking with secondary data).
    bool primary_only = false;

    /// Default constructor (for deserialization).
    IndexSVSVamanaSSD();

    /// Construct and load from a saved directory.
    ///
    /// @param d                    Vector dimensionality.
    /// @param metric               Distance metric (L2 or IP).
    /// @param storage              Storage kind (FP32, LVQ4x8, LeanVec4x8, etc.).
    /// @param saved_dir            Path to directory with config/, graph/, data/.
    /// @param ssd_path             Path to SSD mount for mmap (empty = RAM mode).
    /// @param primary_placement    Where to place primary data (RAM or SSD).
    /// @param secondary_placement  Where to place secondary/residual data (RAM or SSD).
    /// @param search_window        Search window size.
    /// @param search_buffer        Search buffer capacity.
    /// @param primary_only         Use primary-only mode (LeanVec only).
    IndexSVSVamanaSSD(
            idx_t d,
            MetricType metric,
            SVSStorageKind storage,
            const char* saved_dir,
            const char* ssd_path = "",
            SVSDataPlacement primary_placement = SVS_PLACEMENT_RAM,
            SVSDataPlacement secondary_placement = SVS_PLACEMENT_RAM,
            size_t search_window = 10,
            size_t search_buffer = 10,
            bool primary_only = false);

    ~IndexSVSVamanaSSD() override;

    /// Not supported — this is a read-only index.
    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    void reset() override;

    /// The static SVS VamanaIndex implementation (read-only).
    svs_runtime::VamanaIndex* impl{nullptr};

    /// Evict all mmap'd pages from memory using madvise(MADV_DONTNEED).
    ///
    /// For SSD-placed data, this discards resident pages so the next
    /// access re-faults from disk.  Useful for benchmarking cold-cache
    /// behaviour — drop_caches alone cannot evict pages that are still
    /// mapped into the process address space.
    static void evict_mmap_pages();

    /// Save an existing IndexSVSVamana (dynamic) index to a directory,
    /// so it can later be loaded as an SSD index via IndexSVSVamanaSSD.
    ///
    /// @param src       The source index to save (must have impl initialized).
    /// @param directory Path to the output directory (must exist).
    static void save_index_to_directory(
            const IndexSVSVamana& src,
            const char* directory);

   private:
    void assemble();
};

} // namespace faiss
