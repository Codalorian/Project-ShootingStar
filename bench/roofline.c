/*
 * ShootingStar Phase 0 -- hardware roofline.
 *
 * Measures the two numbers that bound single-stream LLM decoding:
 *   1. sustained memory READ bandwidth  (weights are streamed once per token)
 *   2. sustained peak arithmetic        (fp32 FMA and int8 VNNI)
 *
 * The ratio of the two is the machine balance in ops/byte. Any decode-time
 * optimisation that does not move the achieved ops/byte toward that balance
 * is not buying wall-clock time on this machine.
 *
 * Emits JSON on stdout, human notes on stderr.
 * Build: make -C bench
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

/* Sinks: keep the optimiser honest. */
volatile double g_sink_d = 0.0;
volatile long   g_sink_i = 0;

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

typedef struct { double median, min, max; } stat_t;

static stat_t summarize(double *v, int n) {
    qsort(v, n, sizeof(double), cmp_double);
    stat_t s;
    s.min = v[0];
    s.max = v[n - 1];
    s.median = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    return s;
}

/* ------------------------------------------------------------------ */
/* Sequential read bandwidth. This is the LLM weight-streaming pattern. */
/* ------------------------------------------------------------------ */
static double read_bw_once(float *buf, size_t n, int nthreads) {
    double t0 = omp_get_wtime();
    double total = 0.0;
#pragma omp parallel num_threads(nthreads) reduction(+ : total)
    {
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
#pragma omp for schedule(static)
        for (size_t i = 0; i < n; i += 64) {
            a0 = _mm512_add_ps(a0, _mm512_load_ps(buf + i));
            a1 = _mm512_add_ps(a1, _mm512_load_ps(buf + i + 16));
            a2 = _mm512_add_ps(a2, _mm512_load_ps(buf + i + 32));
            a3 = _mm512_add_ps(a3, _mm512_load_ps(buf + i + 48));
        }
        a0 = _mm512_add_ps(a0, a1);
        a2 = _mm512_add_ps(a2, a3);
        total += (double)_mm512_reduce_add_ps(_mm512_add_ps(a0, a2));
    }
    double t1 = omp_get_wtime();
    g_sink_d += total;
    return (double)(n * sizeof(float)) / (t1 - t0) / 1e9; /* GB/s */
}


/* Cache-tier sweep. The repeat loop lives INSIDE the parallel region: at
 * L1/L2 sizes one omp fork-join (~5-10us) costs more than the kernel itself
 * and would report the cache as slower than DRAM. */
static double read_bw_loops(float *buf, size_t n, int nthreads, int loops) {
    double t0 = omp_get_wtime();
    double total = 0.0;
#pragma omp parallel num_threads(nthreads) reduction(+ : total)
    {
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        for (int l = 0; l < loops; l++) {
#pragma omp for schedule(static)
            for (size_t i = 0; i < n; i += 64) {
                a0 = _mm512_add_ps(a0, _mm512_load_ps(buf + i));
                a1 = _mm512_add_ps(a1, _mm512_load_ps(buf + i + 16));
                a2 = _mm512_add_ps(a2, _mm512_load_ps(buf + i + 32));
                a3 = _mm512_add_ps(a3, _mm512_load_ps(buf + i + 48));
            }
        }
        a0 = _mm512_add_ps(a0, a1);
        a2 = _mm512_add_ps(a2, a3);
        total += (double)_mm512_reduce_add_ps(_mm512_add_ps(a0, a2));
    }
    double t1 = omp_get_wtime();
    g_sink_d += total;
    return (double)(n * sizeof(float)) * loops / (t1 - t0) / 1e9;
}

/* Classic STREAM triad, for comparison with published numbers. */
static double triad_bw_once(float *a, float *b, float *c, size_t n, int nthreads) {
    const float s = 1.0000001f;
    double t0 = omp_get_wtime();
#pragma omp parallel for num_threads(nthreads) schedule(static)
    for (size_t i = 0; i < n; i++) a[i] = b[i] + s * c[i];
    double t1 = omp_get_wtime();
    g_sink_d += a[n / 2];
    /* triad moves 3 arrays: 2 read + 1 write */
    return (double)(3 * n * sizeof(float)) / (t1 - t0) / 1e9;
}

/* ------------------------------------------------------------------ */
/* Peak arithmetic: dependency-free chains, cache-resident.            */
/* ------------------------------------------------------------------ */
#define NACC 12

static double fp32_fma_gflops(long iters, int nthreads) {
    double t0 = omp_get_wtime();
    double sink = 0.0;
#pragma omp parallel num_threads(nthreads) reduction(+ : sink)
    {
        __m512 acc[NACC];
        __m512 x = _mm512_set1_ps(1.0000001f);
        __m512 y = _mm512_set1_ps(0.9999999f);
        for (int j = 0; j < NACC; j++) acc[j] = _mm512_set1_ps((float)j);
        for (long i = 0; i < iters; i++) {
#pragma GCC unroll 12
            for (int j = 0; j < NACC; j++) acc[j] = _mm512_fmadd_ps(x, y, acc[j]);
        }
        __m512 s = acc[0];
        for (int j = 1; j < NACC; j++) s = _mm512_add_ps(s, acc[j]);
        sink += (double)_mm512_reduce_add_ps(s);
    }
    double t1 = omp_get_wtime();
    g_sink_d += sink;
    /* 16 lanes * 2 flops per FMA */
    double flops = (double)iters * NACC * 16.0 * 2.0 * nthreads;
    return flops / (t1 - t0) / 1e9;
}

static double int8_vnni_gops(long iters, int nthreads) {
    double t0 = omp_get_wtime();
    long sink = 0;
#pragma omp parallel num_threads(nthreads) reduction(+ : sink)
    {
        __m512i acc[NACC];
        __m512i u = _mm512_set1_epi8((char)3);   /* unsigned operand */
        __m512i v = _mm512_set1_epi8((char)5);   /* signed operand   */
        for (int j = 0; j < NACC; j++) acc[j] = _mm512_setzero_si512();
        for (long i = 0; i < iters; i++) {
#pragma GCC unroll 12
            for (int j = 0; j < NACC; j++) acc[j] = _mm512_dpbusd_epi32(acc[j], u, v);
        }
        __m512i s = acc[0];
        for (int j = 1; j < NACC; j++) s = _mm512_add_epi32(s, acc[j]);
        sink += (long)_mm512_reduce_add_epi32(s);
    }
    double t1 = omp_get_wtime();
    g_sink_i += sink;
    /* dpbusd: 64 int8 multiplies + 64 adds per instruction */
    double ops = (double)iters * NACC * 128.0 * nthreads;
    return ops / (t1 - t0) / 1e9;
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv) {
    int reps = (argc > 1) ? atoi(argv[1]) : 7;
    size_t dram_bytes = 1024UL * 1024 * 1024; /* 1 GiB, >> 12 MiB L3 */
    size_t n = dram_bytes / sizeof(float);
    int max_threads = omp_get_max_threads();

    fprintf(stderr, "[roofline] max_threads=%d reps=%d dram_buf=%.2f GiB\n",
            max_threads, reps, dram_bytes / 1073741824.0);

    float *a = aligned_alloc(64, n * sizeof(float));
    float *b = aligned_alloc(64, n * sizeof(float));
    float *c = aligned_alloc(64, n * sizeof(float));
    if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* First touch in parallel with the same schedule the kernels use. */
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; c[i] = 3.0f; }

    double *buf = malloc(reps * sizeof(double));
    printf("{\n  \"reps\": %d,\n  \"max_threads\": %d,\n", reps, max_threads);

    /* --- read bandwidth vs thread count --- */
    printf("  \"read_bw_gbps\": {");
    int tcs[] = {1, 2, 4, 8};
    for (int k = 0; k < 4; k++) {
        int t = tcs[k];
        if (t > max_threads) continue;
        read_bw_once(b, n, t); /* warm */
        for (int r = 0; r < reps; r++) buf[r] = read_bw_once(b, n, t);
        stat_t s = summarize(buf, reps);
        printf("%s\n    \"%d\": {\"median\": %.2f, \"min\": %.2f, \"max\": %.2f}",
               k ? "," : "", t, s.median, s.min, s.max);
        fprintf(stderr, "[roofline] read  %d thr: %6.2f GB/s (min %.2f max %.2f)\n",
                t, s.median, s.min, s.max);
    }
    printf("\n  },\n");

    /* --- cache hierarchy sweep, single thread and all threads --- */
    printf("  \"read_bw_size_sweep_gbps\": {");
    size_t sizes[] = {16UL<<10, 128UL<<10, 1UL<<20, 4UL<<20, 12UL<<20,
                      32UL<<20, 256UL<<20, 1024UL<<20};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    for (int k = 0; k < nsizes; k++) {
        size_t sn = sizes[k] / sizeof(float);
        if (sn < 64) continue;
        /* aim for >=50 ms of work per measurement regardless of size */
        int loops = (int)(2000000000.0 / (double)sizes[k]);
        if (loops < 3) loops = 3;
        if (loops > 20000) loops = 20000;
        read_bw_loops(b, sn, max_threads, loops > 8 ? 8 : loops);
        for (int r = 0; r < reps; r++) buf[r] = read_bw_loops(b, sn, max_threads, loops);
        stat_t s = summarize(buf, reps);
        double inner = s.median;
        printf("%s\n    \"%zu\": %.2f", k ? "," : "", sizes[k], inner);
        fprintf(stderr, "[roofline] read %7zu KiB (%d thr): %8.2f GB/s\n",
                sizes[k] >> 10, max_threads, inner);
    }
    printf("\n  },\n");

    /* --- triad --- */
    triad_bw_once(a, b, c, n, max_threads);
    for (int r = 0; r < reps; r++) buf[r] = triad_bw_once(a, b, c, n, max_threads);
    stat_t tr = summarize(buf, reps);
    printf("  \"triad_bw_gbps_allthreads\": {\"median\": %.2f, \"min\": %.2f, \"max\": %.2f},\n",
           tr.median, tr.min, tr.max);
    fprintf(stderr, "[roofline] triad %d thr: %6.2f GB/s\n", max_threads, tr.median);

    /* --- compute peaks --- */
    long fma_iters = 2000000;
    printf("  \"fp32_fma_gflops\": {");
    for (int k = 0; k < 4; k++) {
        int t = tcs[k];
        if (t > max_threads) continue;
        fp32_fma_gflops(fma_iters / 10, t);
        for (int r = 0; r < reps; r++) buf[r] = fp32_fma_gflops(fma_iters, t);
        stat_t s = summarize(buf, reps);
        printf("%s\n    \"%d\": {\"median\": %.2f, \"min\": %.2f, \"max\": %.2f}",
               k ? "," : "", t, s.median, s.min, s.max);
        fprintf(stderr, "[roofline] fp32 FMA %d thr: %8.2f GFLOP/s\n", t, s.median);
    }
    printf("\n  },\n");

    printf("  \"int8_vnni_gops\": {");
    for (int k = 0; k < 4; k++) {
        int t = tcs[k];
        if (t > max_threads) continue;
        int8_vnni_gops(fma_iters / 10, t);
        for (int r = 0; r < reps; r++) buf[r] = int8_vnni_gops(fma_iters, t);
        stat_t s = summarize(buf, reps);
        printf("%s\n    \"%d\": {\"median\": %.2f, \"min\": %.2f, \"max\": %.2f}",
               k ? "," : "", t, s.median, s.min, s.max);
        fprintf(stderr, "[roofline] int8 VNNI %d thr: %8.2f GOP/s\n", t, s.median);
    }
    printf("\n  }\n}\n");

    free(a); free(b); free(c); free(buf);
    return 0;
}
