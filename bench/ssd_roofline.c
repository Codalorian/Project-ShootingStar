/* ShootingStar Phase 0 — storage roofline.
 *
 * Measures what the NVMe can actually deliver under the access patterns a
 * block loader would issue: O_DIRECT reads, swept over block size and queue
 * depth, sequential and random.
 *
 * O_DIRECT is the point. Buffered reads land in the page cache, which on a
 * 31 GB machine competes with the block cache we intend to manage ourselves,
 * and makes every measurement a lie about RAM rather than a fact about flash.
 *
 * Queue depth is modelled with threads: QD = concurrent in-flight preads.
 * No liburing dependency; io_uring can replace this if syscall overhead shows
 * up as a ceiling (it will not below ~500k IOPS).
 *
 * Build:  make -C bench
 * Run:    bench/ssd_roofline --file /path/to/testfile --size 32
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>
#include <stdint.h>
#include <sys/stat.h>

#define ALIGN      4096
#define MAX_LAT    20000
#define MAX_THREAD 256

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* xorshift64: rand_r is 31-bit and we index block counts beyond that. */
static inline uint64_t xs64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return (*s = x);
}

typedef struct {
    int      fd;
    size_t   bs;
    off_t    filesize;
    int      seq;
    int      tid;
    int      nthreads;
    double   duration;
    uint64_t bytes;
    uint64_t ops;
    uint64_t errors;
    double   lat_sum;
    double   lat[MAX_LAT];
    int      nlat;
    uint64_t seed;
} worker_t;

static void *worker(void *arg) {
    worker_t *w = arg;
    void *buf = NULL;
    if (posix_memalign(&buf, ALIGN, w->bs) != 0) { w->errors++; return NULL; }

    off_t region = w->filesize / w->nthreads;
    off_t base   = region * (off_t)w->tid;
    off_t pos    = 0;
    uint64_t nblocks = (uint64_t)(w->filesize / (off_t)w->bs);
    if (nblocks == 0) nblocks = 1;

    double t0 = now(), t;
    while ((t = now()) - t0 < w->duration) {
        off_t off;
        if (w->seq) {
            if (pos + (off_t)w->bs > region) pos = 0;
            off = base + pos;
            pos += (off_t)w->bs;
        } else {
            off = (off_t)(xs64(&w->seed) % nblocks) * (off_t)w->bs;
        }
        double s = now();
        ssize_t r = pread(w->fd, buf, w->bs, off);
        double e = now();
        if (r != (ssize_t)w->bs) { w->errors++; if (w->errors > 64) break; continue; }
        w->bytes += (uint64_t)r;
        w->ops++;
        double l = e - s;
        w->lat_sum += l;
        if (w->nlat < MAX_LAT) w->lat[w->nlat++] = l;
    }
    free(buf);
    return NULL;
}

static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

typedef struct {
    size_t bs; int qd; int seq;
    double gbps, iops, lat_mean, lat_p50, lat_p99, lat_max;
    uint64_t errors;
} result_t;

static result_t run_one(const char *path, size_t bs, int qd, int seq, double dur) {
    result_t res; memset(&res, 0, sizeof res);
    res.bs = bs; res.qd = qd; res.seq = seq;

    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); res.errors = 1; return res; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); res.errors = 1; return res; }

    static worker_t w[MAX_THREAD];
    static pthread_t th[MAX_THREAD];
    memset(w, 0, sizeof(worker_t) * (size_t)qd);

    double t0 = now();
    for (int i = 0; i < qd; i++) {
        w[i].fd = fd; w[i].bs = bs; w[i].filesize = st.st_size;
        w[i].seq = seq; w[i].tid = i; w[i].nthreads = qd;
        w[i].duration = dur;
        w[i].seed = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)i * 0xBF58476D1CE4E5B9ULL) ^ (uint64_t)(t0 * 1e6);
        if (w[i].seed == 0) w[i].seed = 1;
        pthread_create(&th[i], NULL, worker, &w[i]);
    }
    for (int i = 0; i < qd; i++) pthread_join(th[i], NULL);
    double elapsed = now() - t0;
    close(fd);

    uint64_t bytes = 0, ops = 0; double lsum = 0;
    static double all[MAX_LAT * 8]; int nall = 0;
    for (int i = 0; i < qd; i++) {
        bytes += w[i].bytes; ops += w[i].ops; lsum += w[i].lat_sum;
        res.errors += w[i].errors;
        for (int j = 0; j < w[i].nlat && nall < (int)(sizeof all / sizeof all[0]); j++)
            all[nall++] = w[i].lat[j];
    }
    if (elapsed <= 0 || ops == 0) return res;
    res.gbps = (double)bytes / elapsed / 1e9;
    res.iops = (double)ops / elapsed;
    res.lat_mean = lsum / (double)ops;
    if (nall > 0) {
        qsort(all, (size_t)nall, sizeof(double), cmpd);
        res.lat_p50 = all[nall / 2];
        res.lat_p99 = all[(int)((double)nall * 0.99)];
        res.lat_max = all[nall - 1];
    }
    return res;
}

static void human_bs(size_t bs, char *out, size_t n) {
    if (bs >= (1u << 20)) snprintf(out, n, "%zuM", bs >> 20);
    else                  snprintf(out, n, "%zuK", bs >> 10);
}

static int make_testfile(const char *path, double gib) {
    struct stat st;
    off_t want = (off_t)(gib * 1073741824.0);
    if (stat(path, &st) == 0 && st.st_size >= want) {
        fprintf(stderr, "test file exists (%.1f GiB), reusing\n", (double)st.st_size / 1073741824.0);
        return 0;
    }
    fprintf(stderr, "creating %.0f GiB test file at %s ...\n", gib, path);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("create"); return -1; }
    size_t chunk = 8u << 20;
    char *buf = malloc(chunk);
    if (!buf) { close(fd); return -1; }
    /* Real data, not zeros: sparse/zero extents can be served without touching
     * flash and would flatter every number in the sweep. */
    uint64_t s = 0x123456789ABCDEFULL;
    for (size_t i = 0; i < chunk; i += 8) *(uint64_t *)(buf + i) = xs64(&s);
    off_t written = 0;
    double t0 = now();
    while (written < want) {
        ssize_t r = write(fd, buf, chunk);
        if (r <= 0) { perror("write"); free(buf); close(fd); return -1; }
        written += r;
        if ((written >> 20) % 4096 == 0)
            fprintf(stderr, "\r  %.1f / %.0f GiB", (double)written / 1073741824.0, gib);
    }
    fsync(fd);
    fprintf(stderr, "\r  wrote %.0f GiB in %.1fs\n", gib, now() - t0);
    free(buf); close(fd);
    return 0;
}

int main(int argc, char **argv) {
    const char *path = "ssd_roofline.dat";
    const char *json = "results/ssd_roofline.json";
    double size_gib = 32.0, dur = 2.0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--file") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) size_gib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--dur") && i + 1 < argc) dur = atof(argv[++i]);
        else if (!strcmp(argv[i], "--json") && i + 1 < argc) json = argv[++i];
        else { fprintf(stderr, "usage: %s [--file F] [--size GiB] [--dur S] [--json F]\n", argv[0]); return 2; }
    }
    if (make_testfile(path, size_gib) != 0) return 1;

    size_t bss[] = { 4096, 16384, 65536, 131072, 262144, 1u<<20, 4u<<20 };
    int    qds[] = { 1, 2, 4, 8, 16, 32, 64 };
    int nbs = sizeof bss / sizeof bss[0], nqd = sizeof qds / sizeof qds[0];

    enum { MAX_RESULTS = 2 * 7 * 7 + 7 };  /* sweep + QD1 latency profile */
    static result_t all[MAX_RESULTS]; int n = 0;
    printf("\nShootingStar storage roofline — O_DIRECT, QD via threads\n");
    printf("device: reads bypass page cache; file %.0f GiB; %.1fs per point\n\n", size_gib, dur);

    for (int seq = 1; seq >= 0; seq--) {
        printf("=== %s ===\n", seq ? "SEQUENTIAL" : "RANDOM");
        printf("%6s", "bs\\qd");
        for (int q = 0; q < nqd; q++) printf("%9d", qds[q]);
        printf("     (GB/s)\n");
        for (int b = 0; b < nbs; b++) {
            char hb[16]; human_bs(bss[b], hb, sizeof hb);
            printf("%6s", hb);
            fflush(stdout);
            for (int q = 0; q < nqd; q++) {
                result_t r = run_one(path, bss[b], qds[q], seq, dur);
                if (n < MAX_RESULTS) all[n++] = r;
                printf("%9.2f", r.gbps);
                fflush(stdout);
            }
            printf("\n");
        }
        printf("\n");
    }

    /* QD1 latency profile — the number that prices a cache miss on the
     * critical path. */
    printf("=== QD1 LATENCY (random, the cache-miss cost) ===\n");
    printf("%8s %10s %10s %10s %10s\n", "bs", "mean_us", "p50_us", "p99_us", "max_us");
    for (int b = 0; b < nbs; b++) {
        result_t r = run_one(path, bss[b], 1, 0, dur);
        char hb[16]; human_bs(bss[b], hb, sizeof hb);
        printf("%8s %10.1f %10.1f %10.1f %10.1f\n", hb,
               r.lat_mean*1e6, r.lat_p50*1e6, r.lat_p99*1e6, r.lat_max*1e6);
        if (n < MAX_RESULTS) all[n++] = r;
    }

    FILE *f = fopen(json, "w");
    if (f) {
        fprintf(f, "{\n  \"file_gib\": %.1f,\n  \"dur_s\": %.1f,\n  \"points\": [\n", size_gib, dur);
        for (int i = 0; i < n; i++) {
            fprintf(f, "    {\"bs\": %zu, \"qd\": %d, \"pattern\": \"%s\", \"gbps\": %.4f, "
                       "\"iops\": %.1f, \"lat_mean_us\": %.2f, \"lat_p50_us\": %.2f, "
                       "\"lat_p99_us\": %.2f, \"lat_max_us\": %.2f, \"errors\": %llu}%s\n",
                    all[i].bs, all[i].qd, all[i].seq ? "seq" : "rand", all[i].gbps,
                    all[i].iops, all[i].lat_mean*1e6, all[i].lat_p50*1e6,
                    all[i].lat_p99*1e6, all[i].lat_max*1e6,
                    (unsigned long long)all[i].errors, i == n-1 ? "" : ",");
        }
        fprintf(f, "  ]\n}\n");
        fclose(f);
        fprintf(stderr, "\nwrote %s\n", json);
    } else {
        fprintf(stderr, "\ncould not write %s: %s\n", json, strerror(errno));
    }
    return 0;
}
