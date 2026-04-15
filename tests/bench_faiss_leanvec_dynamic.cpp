/*
 * Dynamic LeanVec primary-only benchmark.
 *
 * 1. Train + build with primary_only=true
 * 2. Search and report recall/QPS/memory
 * 3. Save (faiss::write_index) and load (faiss::read_index)
 * 4. Search again on the loaded index
 * 5. Add new vectors, search again
 * 6. Delete some vectors, search again
 *
 * Build (from faiss_fork/build):
 *   make -j bench_faiss_leanvec_dynamic
 *
 * Run:
 *   LD_LIBRARY_PATH=<svs_runtime_lib_dir> \
 *       taskset -c 56 numactl -m 1 \
 *       ./tests/bench_faiss_leanvec_dynamic [SW] [base] [queries] [gt]
 */

#include <faiss/Index.h>
#include <faiss/index_io.h>
#include <faiss/svs/IndexSVSVamanaLeanVec.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

// ----------------------------------------------------------------
// .fvecs / .ivecs helpers
// ----------------------------------------------------------------
static std::vector<float> fvecs_read(const char* fname, int& d_out, int& n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname); exit(1); }
    int d;
    if (fread(&d, sizeof(int), 1, f) != 1) { fclose(f); exit(1); }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = file_size / ((d + 1) * sizeof(float));
    d_out = d; n_out = n;
    std::vector<float> data(static_cast<size_t>(n) * d);
    std::vector<float> row(d + 1);
    for (int i = 0; i < n; i++) {
        if (fread(row.data(), sizeof(float), d + 1, f) != static_cast<size_t>(d + 1))
            break;
        std::copy(row.begin() + 1, row.end(), data.begin() + static_cast<size_t>(i) * d);
    }
    fclose(f);
    return data;
}

static std::vector<int> ivecs_read(const char* fname, int& d_out, int& n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname); exit(1); }
    int d;
    if (fread(&d, sizeof(int), 1, f) != 1) { fclose(f); exit(1); }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = file_size / ((d + 1) * sizeof(int));
    d_out = d; n_out = n;
    std::vector<int> data(static_cast<size_t>(n) * d);
    std::vector<int> row(d + 1);
    for (int i = 0; i < n; i++) {
        if (fread(row.data(), sizeof(int), d + 1, f) != static_cast<size_t>(d + 1))
            break;
        std::copy(row.begin() + 1, row.end(), data.begin() + static_cast<size_t>(i) * d);
    }
    fclose(f);
    return data;
}

// ----------------------------------------------------------------
// Memory helpers
// ----------------------------------------------------------------
struct MemStats {
    long rss_anon_kb = 0, rss_file_kb = 0;
    static MemStats read() {
        MemStats s;
        FILE* f = fopen("/proc/self/status", "r");
        if (!f) return s;
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "RssAnon:", 8) == 0) sscanf(line + 8, "%ld", &s.rss_anon_kb);
            else if (strncmp(line, "RssFile:", 8) == 0) sscanf(line + 8, "%ld", &s.rss_file_kb);
        }
        fclose(f);
        return s;
    }
};

// ----------------------------------------------------------------
// Recall
// ----------------------------------------------------------------
static double recall_at_k(
        const std::vector<int>& gt, const std::vector<faiss::idx_t>& pred,
        int nq, int gt_k, int k) {
    double total = 0;
    for (int q = 0; q < nq; q++) {
        int found = 0;
        for (int i = 0; i < k; i++) {
            auto pid = pred[q * k + i];
            for (int j = 0; j < k; j++) {
                if (pid == static_cast<faiss::idx_t>(gt[q * gt_k + j])) {
                    ++found; break;
                }
            }
        }
        total += static_cast<double>(found) / k;
    }
    return total / nq;
}

// ----------------------------------------------------------------
// Percentile
// ----------------------------------------------------------------
static double percentile(std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0;
    double idx = p / 100.0 * (sorted.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = lo + 1;
    if (hi >= sorted.size()) return sorted.back();
    double frac = idx - lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

// ----------------------------------------------------------------
// Search + report
// ----------------------------------------------------------------
static void run_search(
        const char* label,
        faiss::IndexSVSVamanaLeanVec* index,
        const float* queries, int nq, int d, int k,
        const std::vector<int>& gt, int gt_k,
        size_t search_window, size_t num_reps,
        const float* base = nullptr) {

    faiss::SearchParametersSVSVamana sp;
    sp.search_window_size = search_window;
    sp.search_buffer_capacity = search_window;

    // Retrieve search_window candidates so we can rerank
    int k_retrieve = static_cast<int>(search_window);
    std::vector<float> dists(nq * k_retrieve);
    std::vector<faiss::idx_t> labels_all(nq * k_retrieve);
    std::vector<double> qps_values;
    std::vector<double> batch_latencies;

    for (size_t rep = 0; rep < num_reps; rep++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        index->search(nq, queries, k_retrieve, dists.data(), labels_all.data(), &sp);
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        qps_values.push_back(nq / dt);
        batch_latencies.push_back(dt);
    }

    // Top-k from primary-only distances
    std::vector<faiss::idx_t> labels_topk(nq * k);
    for (int q = 0; q < nq; q++) {
        for (int i = 0; i < k; i++) {
            labels_topk[q * k + i] = labels_all[q * k_retrieve + i];
        }
    }
    double rec = recall_at_k(gt, labels_topk, nq, gt_k, k);

    // Rerank using original fp32 base vectors
    double rec_rerank = 0;
    if (base) {
        std::vector<faiss::idx_t> labels_reranked(nq * k);
        std::vector<std::pair<float, faiss::idx_t>> cands(k_retrieve);

        for (int q = 0; q < nq; q++) {
            const float* qvec = queries + static_cast<size_t>(q) * d;
            for (int c = 0; c < k_retrieve; c++) {
                faiss::idx_t id = labels_all[q * k_retrieve + c];
                float dist = 0;
                if (id >= 0) {
                    const float* bvec = base + static_cast<size_t>(id) * d;
                    for (int j = 0; j < d; j++) {
                        float diff = qvec[j] - bvec[j];
                        dist += diff * diff;
                    }
                } else {
                    dist = std::numeric_limits<float>::max();
                }
                cands[c] = {dist, id};
            }
            std::partial_sort(cands.begin(), cands.begin() + k, cands.end());
            for (int i = 0; i < k; i++) {
                labels_reranked[q * k + i] = cands[i].second;
            }
        }
        rec_rerank = recall_at_k(gt, labels_reranked, nq, gt_k, k);
    }

    double qps_max = *std::max_element(qps_values.begin(), qps_values.end());
    std::sort(batch_latencies.begin(), batch_latencies.end());
    double lat_avg = std::accumulate(batch_latencies.begin(), batch_latencies.end(), 0.0)
                     / batch_latencies.size();
    double lat_p50 = percentile(batch_latencies, 50);
    double lat_p99 = percentile(batch_latencies, 99);
    auto mem = MemStats::read();

    if (base) {
        printf("%-22s | ntotal=%7ld | %.4f | %.4f | %7.0f | %7.3f %7.3f %7.3f | %6.0fMB %6.0fMB\n",
               label, static_cast<long>(index->ntotal), rec, rec_rerank, qps_max,
               lat_avg * 1000, lat_p50 * 1000, lat_p99 * 1000,
               mem.rss_anon_kb / 1024.0, mem.rss_file_kb / 1024.0);
    } else {
        printf("%-22s | ntotal=%7ld | %.4f |    N/A | %7.0f | %7.3f %7.3f %7.3f | %6.0fMB %6.0fMB\n",
               label, static_cast<long>(index->ntotal), rec, qps_max,
               lat_avg * 1000, lat_p50 * 1000, lat_p99 * 1000,
               mem.rss_anon_kb / 1024.0, mem.rss_file_kb / 1024.0);
    }
    fflush(stdout);
}

// ----------------------------------------------------------------
// Main
// ----------------------------------------------------------------
int main(int argc, char** argv) {
    const char* base_file  = (argc > 1) ? argv[1] : "/path/to/gist/gist_base.fvecs";
    const char* query_file = (argc > 2) ? argv[2] : "/path/to/gist/gist_queries.fvecs";
    const char* gt_file    = (argc > 3) ? argv[3] : "/path/to/gist/gist_gtruth.ivecs";
    size_t search_window   = (argc > 4) ? std::stoul(argv[4]) : 100;
    size_t num_reps        = (argc > 5) ? std::stoul(argv[5]) : 3;

    const int k = 10;
    const size_t degree = 64;

    // Load base vectors
    int bd, bn;
    auto base = fvecs_read(base_file, bd, bn);
    printf("Base: %d x %d\n", bn, bd);

    // Load queries
    int qd, nq;
    auto queries = fvecs_read(query_file, qd, nq);
    printf("Queries: %d x %d\n", nq, qd);

    // Load ground truth
    int gt_d, gt_n;
    auto gt = ivecs_read(gt_file, gt_d, gt_n);
    printf("Ground truth: %d x %d\n", gt_n, gt_d);

    printf("\nSW=%zu  Reps=%zu  k=%d  degree=%zu  primary_only=true\n\n", search_window, num_reps, k, degree);

    // Print header
    printf("%-22s | %13s | %7s | %7s | %7s | %7s %7s %7s | %8s %8s\n",
           "Stage", "ntotal", "Recall", "Rerank", "QPS",
           "avg_ms", "p50_ms", "p99_ms",
           "RssAnon", "RssFile");
    printf("--------------------------------------------------"
           "--------------------------------------------------"
           "---------------------------\n");

    // ============================================================
    // 1. Train + Build with primary_only=true
    // ============================================================
    auto* index = new faiss::IndexSVSVamanaLeanVec(
            bd, degree, faiss::METRIC_L2,
            /*leanvec_dims=*/0,
            faiss::SVS_LeanVec4x8,
            /*primary_only=*/true);

    index->search_window_size = search_window;
    index->search_buffer_capacity = search_window;
    index->construction_window_size = 200;
    index->alpha = 1.2;

    printf("Training...\n"); fflush(stdout);
    index->train(bn, base.data());

    printf("Adding %d vectors...\n", bn); fflush(stdout);
    index->add(bn, base.data());

    // 2. Search after build
    run_search("After build", index, queries.data(), nq, bd, k, gt, gt_d, search_window, num_reps, base.data());

    // ============================================================
    // 3. Save and Load
    // ============================================================
    std::string save_path = "/tmp/bench_leanvec_dynamic_pri.faiss";
    printf("Saving to %s ...\n", save_path.c_str()); fflush(stdout);
    faiss::write_index(index, save_path.c_str());
    delete index;
    index = nullptr;

    printf("Loading from %s ...\n", save_path.c_str()); fflush(stdout);
    auto* loaded_raw = faiss::read_index(save_path.c_str());
    auto* loaded = dynamic_cast<faiss::IndexSVSVamanaLeanVec*>(loaded_raw);
    if (!loaded) {
        printf("ERROR: loaded index is not IndexSVSVamanaLeanVec\n");
        delete loaded_raw;
        return 1;
    }

    // 4. Search after load
    run_search("After load", loaded, queries.data(), nq, bd, k, gt, gt_d, search_window, num_reps, base.data());

    // ============================================================
    // 5. Add new vectors (re-add first 1000 base vectors as new IDs)
    // ============================================================
    size_t n_add = 1000;
    printf("Adding %zu more vectors...\n", n_add); fflush(stdout);
    loaded->add(n_add, base.data());

    run_search("After add", loaded, queries.data(), nq, bd, k, gt, gt_d, search_window, num_reps, base.data());

    // ============================================================
    // 6. Delete some vectors
    // ============================================================
    size_t n_del = 500;
    // Delete the last n_del IDs that were just added
    faiss::idx_t del_start = loaded->ntotal - static_cast<faiss::idx_t>(n_del);
    faiss::IDSelectorRange sel(del_start, loaded->ntotal);
    printf("Deleting %zu vectors (IDs %ld..%ld)...\n",
           n_del, static_cast<long>(del_start),
           static_cast<long>(loaded->ntotal - 1));
    fflush(stdout);
    size_t removed = loaded->remove_ids(sel);
    printf("Removed %zu vectors\n", removed); fflush(stdout);

    run_search("After delete", loaded, queries.data(), nq, bd, k, gt, gt_d, search_window, num_reps, base.data());

    // Cleanup
    delete loaded;
    std::filesystem::remove(save_path);

    printf("\nDone.\n");
    return 0;
}
