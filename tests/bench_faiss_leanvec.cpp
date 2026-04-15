/*
 * Faiss-level LeanVec4x8 benchmark — matches the native/runtime configs.
 *
 * Uses IndexSVSVamanaSSD to load a pre-built PCA index, then benchmarks
 * with per-batch latency (p50/p95/p99), recall@10, and memory stats.
 *
 * Build (from faiss_fork/build):
 *   make -j bench_faiss_leanvec
 *
 * Run:
 *   ./tests/bench_faiss_leanvec [SW] [BS] [reps] [index_dir] [mode]
 */

#include <faiss/svs/IndexSVSVamanaSSD.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include <sys/mman.h>
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
    long rss_kb = 0, rss_anon_kb = 0, rss_file_kb = 0;
    static MemStats read() {
        MemStats s;
        FILE* f = fopen("/proc/self/status", "r");
        if (!f) return s;
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "VmRSS:", 6) == 0) sscanf(line + 6, "%ld", &s.rss_kb);
            else if (strncmp(line, "RssAnon:", 8) == 0) sscanf(line + 8, "%ld", &s.rss_anon_kb);
            else if (strncmp(line, "RssFile:", 8) == 0) sscanf(line + 8, "%ld", &s.rss_file_kb);
        }
        fclose(f);
        return s;
    }
};

struct PFStats {
    long minor = 0, major = 0;
    static PFStats read() {
        struct rusage ru;
        getrusage(RUSAGE_SELF, &ru);
        return {ru.ru_minflt, ru.ru_majflt};
    }
    PFStats delta(const PFStats& before) const {
        return {minor - before.minor, major - before.major};
    }
};

static void evict_mmap_pages() {
    std::ifstream maps("/proc/self/maps");
    if (!maps.is_open()) return;
    std::string line;
    while (std::getline(maps, line)) {
        if (line.size() < 50) continue;
        auto sp = line.find(' ');
        if (sp == std::string::npos) continue;
        std::string perms = line.substr(sp + 1, 4);
        if (perms.size() < 4 || perms[3] != 's') continue;
        auto dash = line.find('-');
        if (dash == std::string::npos || dash >= sp) continue;
        unsigned long start = std::stoul(line.substr(0, dash), nullptr, 16);
        unsigned long end_addr = std::stoul(line.substr(dash + 1, sp - dash - 1), nullptr, 16);
        size_t len = end_addr - start;
        if (len > 0) {
            (void)madvise(reinterpret_cast<void*>(start), len, MADV_DONTNEED);
        }
    }
}

static void drop_caches() {
    evict_mmap_pages();
    sync();
    FILE* f = fopen("/proc/sys/vm/drop_caches", "w");
    if (f) { fprintf(f, "3\n"); fclose(f); }
}

// ----------------------------------------------------------------
// Recall
// ----------------------------------------------------------------
static double recall_at_k(
        const std::vector<int>& gt, const std::vector<faiss::idx_t>& pred,
        int nq, int gt_k, int k, int gt_check = 0) {
    if (gt_check <= 0) gt_check = k;
    double total = 0;
    for (int q = 0; q < nq; q++) {
        int found = 0;
        for (int i = 0; i < k; i++) {
            auto pid = pred[q * k + i];
            for (int j = 0; j < gt_check; j++) {
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
// Config
// ----------------------------------------------------------------
struct Config {
    const char* name;
    faiss::SVSDataPlacement primary;
    faiss::SVSDataPlacement secondary;
    const char* ssd_path;
    bool primary_only;
};

// ----------------------------------------------------------------
// Main
// ----------------------------------------------------------------
int main(int argc, char** argv) {
    const char* query_file = "/path/to/gist/gist_queries.fvecs";
    const char* gt_file    = "/path/to/gist/gist_gtruth.ivecs";
    const char* index_dir  = (argc > 4) ? argv[4] : "/path/to/saved_index";
    const char* ssd_prefix = "/mnt";

    const int d = 960, k = 10, gt_k = 100;

    size_t search_window = (argc > 1) ? std::stoul(argv[1]) : 100;
    size_t batch_size    = (argc > 2) ? std::stoul(argv[2]) : 1;
    size_t num_reps      = (argc > 3) ? std::stoul(argv[3]) : 3;
    std::string mode     = (argc > 5) ? argv[5] : "all";
    bool ram_only        = (mode == "ram");

    printf("=== Faiss LeanVec4x8 Benchmark ===\n");
    printf("Index:   %s\n", index_dir);
    printf("SW=%zu  BS=%zu  Reps=%zu  k=%d\n\n", search_window, batch_size, num_reps, k);

    // Load queries & ground truth
    int qd, nq;
    auto queries = fvecs_read(query_file, qd, nq);
    printf("Queries: %d x %d\n", nq, qd);

    int gt_d, gt_n;
    auto gt = ivecs_read(gt_file, gt_d, gt_n);
    printf("Ground truth: %d x %d\n\n", gt_n, gt_d);

    if (batch_size == 0) batch_size = nq;
    size_t num_batches = (nq + batch_size - 1) / batch_size;

    // Print header
    printf("%-10s | %7s | %7s | %9s %9s %9s %9s %9s | %8s %8s | %s\n",
           "Config", "Recall", "QPS",
           "avg(ms)", "p50(ms)", "p95(ms)", "p99(ms)", "max(ms)",
           "RssAnon", "RssFile", "MajFaults");
    printf("---------------------------------------------"
           "---------------------------------------------"
           "-----------------------------\n");

    // Lambda: run search benchmark on a generic faiss::Index
    auto run_bench = [&](const char* name, faiss::Index* index,
                         const faiss::SearchParameters* sp, bool cold) {
        auto mem_loaded = MemStats::read();

        std::vector<float> dists(nq * k);
        std::vector<faiss::idx_t> labels(nq * k);
        std::vector<double> qps_values;
        std::vector<double> batch_latencies;
        long total_major = 0;

        for (size_t rep = 0; rep < num_reps; rep++) {
            if (cold) drop_caches();

            auto pf0 = PFStats::read();
            double total_time = 0;

            for (size_t bb = 0; bb < num_batches; bb++) {
                size_t start = bb * batch_size;
                size_t end = std::min(start + batch_size, static_cast<size_t>(nq));
                size_t bn = end - start;

                auto t0 = std::chrono::high_resolution_clock::now();
                index->search(
                    bn, queries.data() + start * d, k,
                    dists.data() + start * k,
                    labels.data() + start * k, sp);
                auto t1 = std::chrono::high_resolution_clock::now();
                double dt = std::chrono::duration<double>(t1 - t0).count();
                total_time += dt;
                batch_latencies.push_back(dt);
            }

            auto pf1 = PFStats::read();
            auto pf_d = pf1.delta(pf0);
            total_major += pf_d.major;
            qps_values.push_back(nq / total_time);
        }

        double rec = recall_at_k(gt, labels, nq, gt_k, k, k);
        double qps_max = *std::max_element(qps_values.begin(), qps_values.end());
        std::sort(batch_latencies.begin(), batch_latencies.end());
        double lat_avg = std::accumulate(batch_latencies.begin(), batch_latencies.end(), 0.0)
                         / batch_latencies.size();
        double lat_p50 = percentile(batch_latencies, 50);
        double lat_p95 = percentile(batch_latencies, 95);
        double lat_p99 = percentile(batch_latencies, 99);
        double lat_max = batch_latencies.empty() ? 0 : batch_latencies.back();
        auto mem_final = MemStats::read();

        printf("%-10s | %.4f | %7.0f | %9.3f %9.3f %9.3f %9.3f %9.3f | %6.0fMB %6.0fMB | %ld\n",
               name, rec, qps_max,
               lat_avg * 1000, lat_p50 * 1000, lat_p95 * 1000,
               lat_p99 * 1000, lat_max * 1000,
               mem_loaded.rss_anon_kb / 1024.0,
               mem_final.rss_file_kb / 1024.0,
               total_major);
        fflush(stdout);
    };

    // ---- SSD index modes (IndexSVSVamanaSSD) ----
    Config all_configs[] = {
        {"RAM_RAM",   faiss::SVS_PLACEMENT_RAM, faiss::SVS_PLACEMENT_RAM, "",         false},
        {"RAM_SSD",   faiss::SVS_PLACEMENT_RAM, faiss::SVS_PLACEMENT_SSD, ssd_prefix, false},
        {"SSD_SSD",   faiss::SVS_PLACEMENT_SSD, faiss::SVS_PLACEMENT_SSD, ssd_prefix, false},
    };
    size_t config_start = 0;
    size_t num_configs = 3;
    if (ram_only) {
        num_configs = 1;
    }

    for (size_t ci = config_start; ci < num_configs; ci++) {
        auto& cfg = all_configs[ci];

        drop_caches();

        faiss::IndexSVSVamanaSSD* index = nullptr;
        try {
            index = new faiss::IndexSVSVamanaSSD(
                    d, faiss::METRIC_L2, faiss::SVS_LeanVec4x8,
                    index_dir, cfg.ssd_path,
                    cfg.primary, cfg.secondary,
                    search_window, search_window,
                    cfg.primary_only);
        } catch (const std::exception& e) {
            printf("%-10s | ERROR: %s\n", cfg.name, e.what());
            continue;
        }

        faiss::SearchParametersSVSVamanaSSD sp;
        sp.search_window_size = search_window;
        sp.search_buffer_capacity = search_window;

        bool cold = (cfg.primary == faiss::SVS_PLACEMENT_SSD ||
                     cfg.secondary == faiss::SVS_PLACEMENT_SSD);

        run_bench(cfg.name, index, &sp, cold);
        delete index;
    }

    printf("\nDone.\n");
    return 0;
}
