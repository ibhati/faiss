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

#include <faiss/svs/IndexSVSFaissUtils.h>
#include <faiss/svs/IndexSVSVamanaSSD.h>

#include <faiss/Index.h>

#include <svs/runtime/api_defs.h>
#include <svs/runtime/vamana_index.h>

#include <cstddef>
#include <fstream>
#include <limits>
#include <span>

#include <sys/mman.h>
#include <type_traits>
#include <vector>

namespace faiss {
namespace {
svs_runtime::VamanaIndex::SearchParams make_ssd_search_parameters(
        const IndexSVSVamanaSSD& index,
        const SearchParameters* params) {
    auto search_window_size = index.search_window_size;
    auto search_buffer_capacity = index.search_buffer_capacity;

    if (auto svs_params =
                dynamic_cast<const SearchParametersSVSVamanaSSD*>(params)) {
        if (svs_params->search_window_size > 0)
            search_window_size = svs_params->search_window_size;
        if (svs_params->search_buffer_capacity > 0)
            search_buffer_capacity = svs_params->search_buffer_capacity;
    }

    return {search_window_size, search_buffer_capacity};
}

svs_runtime::SSDConfig make_svs_ssd_config(
        const std::string& ssd_path,
        SVSDataPlacement primary,
        SVSDataPlacement secondary,
        bool primary_only) {
    svs_runtime::SSDConfig config;
    config.ssd_path = ssd_path.empty() ? nullptr : ssd_path.c_str();
    config.primary_on_ssd = (primary == SVS_PLACEMENT_SSD);
    config.secondary_on_ssd = (secondary == SVS_PLACEMENT_SSD);
    config.primary_only = primary_only;
    return config;
}
} // namespace

IndexSVSVamanaSSD::IndexSVSVamanaSSD() = default;

IndexSVSVamanaSSD::IndexSVSVamanaSSD(
        idx_t d,
        MetricType metric,
        SVSStorageKind storage,
        const char* saved_dir,
        const char* ssd_path_str,
        SVSDataPlacement primary_pl,
        SVSDataPlacement secondary_pl,
        size_t search_window,
        size_t search_buffer,
        bool primary_only)
        : Index(d, metric),
          storage_kind{storage},
          search_window_size{search_window},
          search_buffer_capacity{search_buffer},
          saved_directory{saved_dir ? saved_dir : ""},
          ssd_path{ssd_path_str ? ssd_path_str : ""},
          primary_placement{primary_pl},
          secondary_placement{secondary_pl},
          primary_only{primary_only} {
    // Validate the storage kind is available.
    auto svs_storage = to_svs_storage_kind(storage_kind);
    auto status = svs_runtime::VamanaIndex::check_storage_kind(svs_storage);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }

    // Assemble the index from the saved directory.
    assemble();
}

IndexSVSVamanaSSD::~IndexSVSVamanaSSD() {
    if (impl) {
        auto status = svs_runtime::VamanaIndex::destroy(impl);
        FAISS_ASSERT(status.ok());
        impl = nullptr;
    }
}

void IndexSVSVamanaSSD::add(idx_t /*n*/, const float* /*x*/) {
    FAISS_THROW_MSG(
            "IndexSVSVamanaSSD is a read-only index. "
            "Use IndexSVSVamana or IndexSVSVamanaLVQ for mutable indices.");
}

void IndexSVSVamanaSSD::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (!impl) {
        for (idx_t i = 0; i < n * k; ++i) {
            distances[i] = std::numeric_limits<float>::infinity();
            labels[i] = -1;
        }
        return;
    }
    FAISS_THROW_IF_NOT(k > 0);

    auto sp = make_ssd_search_parameters(*this, params);
    auto id_filter = make_faiss_id_filter(params);
    auto status = impl->search(
            static_cast<size_t>(n),
            x,
            static_cast<size_t>(k),
            distances,
            convert_output_buffer<size_t>(labels, static_cast<size_t>(n * k)),
            &sp,
            id_filter.get());

    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
}

void IndexSVSVamanaSSD::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(impl);
    FAISS_THROW_IF_NOT(radius > 0);
    FAISS_THROW_IF_NOT(result->nq == static_cast<size_t>(n));

    auto sp = make_ssd_search_parameters(*this, params);
    auto id_filter = make_faiss_id_filter(params);
    auto status = impl->range_search(
            static_cast<size_t>(n),
            x,
            radius,
            FaissResultsAllocator{result},
            &sp,
            id_filter.get());
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
}

void IndexSVSVamanaSSD::reset() {
    if (impl) {
        auto status = svs_runtime::VamanaIndex::destroy(impl);
        FAISS_ASSERT(status.ok());
        impl = nullptr;
    }
    ntotal = 0;
}

void IndexSVSVamanaSSD::assemble() {
    FAISS_THROW_IF_NOT_MSG(
            !saved_directory.empty(),
            "saved_directory must be set before assembling.");
    FAISS_THROW_IF_NOT_MSG(
            !impl, "Cannot assemble: index already loaded.");

    auto svs_metric = to_svs_metric(metric_type);
    auto svs_storage_kind = to_svs_storage_kind(storage_kind);
    auto svs_ssd_config = make_svs_ssd_config(
            ssd_path, primary_placement, secondary_placement, primary_only);
    auto search_params = svs_runtime::VamanaIndex::SearchParams{
            .search_window_size = search_window_size,
            .search_buffer_capacity = search_buffer_capacity,
    };

    auto status = svs_runtime::VamanaIndex::assemble_from_directory(
            &impl,
            saved_directory.c_str(),
            svs_metric,
            svs_storage_kind,
            svs_ssd_config,
            search_params);

    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
    FAISS_THROW_IF_NOT(impl);

    // The index is trained and ready for search.
    is_trained = true;

    // Query the number of vectors from the first search result
    // or trust the assembled index. For now, we don't have a way to get
    // ntotal from the static VamanaIndex, so we leave it at 0 and document
    // that ntotal may not be accurate for SSD indices.
    // TODO: Add a size() or ntotal() method to VamanaIndex.
}

void IndexSVSVamanaSSD::evict_mmap_pages() {
#ifdef __linux__
    // Parse /proc/self/maps and call MADV_DONTNEED on all file-backed
    // shared mappings. This forces pages out of the process page tables
    // so next access re-faults from disk.
    std::ifstream maps("/proc/self/maps");
    if (!maps.is_open()) return;

    std::string line;
    while (std::getline(maps, line)) {
        // Format: start-end perms offset dev inode pathname
        // Only target shared (s) file-backed mappings (has a pathname)
        if (line.size() < 50) continue;
        // Check for 's' (shared) in the perms field
        auto first_space = line.find(' ');
        if (first_space == std::string::npos) continue;
        std::string perms = line.substr(first_space + 1, 4);
        if (perms.size() < 4 || perms[3] != 's') continue;

        // Parse start-end addresses
        auto dash = line.find('-');
        if (dash == std::string::npos || dash >= first_space) continue;
        unsigned long start = std::stoul(line.substr(0, dash), nullptr, 16);
        unsigned long end = std::stoul(
                line.substr(dash + 1, first_space - dash - 1), nullptr, 16);

        size_t len = end - start;
        if (len > 0) {
            (void)madvise(
                    reinterpret_cast<void*>(start), len, MADV_DONTNEED);
        }
    }
#endif
}

void IndexSVSVamanaSSD::save_index_to_directory(
        const IndexSVSVamana& src,
        const char* directory) {
    FAISS_THROW_IF_NOT_MSG(src.impl, "Source index not initialized.");
    FAISS_THROW_IF_NOT_MSG(directory, "directory must not be null.");

    auto status = src.impl->save_to_directory(directory);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
}

} // namespace faiss
