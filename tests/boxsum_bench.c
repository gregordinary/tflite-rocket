// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The tflite-rocket authors
//
// boxsum_bench — host microbench + cross-mode bit-identity gate for the native-uint8
// direct-conv per-pixel box-sum (rocket_in_window_sum_i8_band, rocket_convert.h).
//
// The box-sum Sx[oh,ow] = sum over the IC*KH*KW conv window of the recentered int8 input
// is the position-dependent correction the asymmetric weight zero-point needs. The
// contiguous-column case (sx==dx==1) has three NEON realizations, selected by
// ROCKET_BOXSUM_MODE (read once per process):
//   0 = separable  (default): channel+row reduce -> horizontal KW window-sum
//   1 = per-window           : the prior form (one 16-wide pass per (ic,kh,kw) tap row)
//   2 = scalar               : the portable reference (also the non-NEON path)
// All three are pure integer adds over the same taps, so they MUST be bit-identical.
//
// Because the mode is cached in a per-process static, this bench fork()s a child per mode
// (each child reads ROCKET_BOXSUM_MODE fresh on its first call), times the real header
// function over representative shapes, and reports a checksum + per-shape timing back to
// the parent over a pipe. The parent fails (exit 1) if any checksum diverges; otherwise it
// prints separable-vs-per-window / vs-scalar speedups. No NPU needed (pure host code), so
// this gate runs identically off-device — but the timings are only meaningful on the RK1
// (the NEON paths compile only under __ARM_NEON).
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>

#include "rocket_convert.h"

static int64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

struct shape { int IC, KH, KW, OH, OW; const char *name; };

/* contiguous (sx=dx=1) native-uint8 direct convs that exercise the box-sum */
static const struct shape shapes[] = {
    {  64, 3, 3, 16, 16, "3x3 IC64  16x16" },
    { 128, 3, 3, 32, 32, "3x3 IC128 32x32" },
    {  48, 5, 5, 16, 16, "5x5 IC48  16x16" },
    {  32, 7, 7, 17, 17, "7x7 IC32  17x17" },
    { 768, 3, 3, 18, 18, "3x3 IC768 18x18" },
    { 768, 1, 1, 20, 20, "1x1 IC768 20x20 (channel reduce)" },
};
#define NSHAPE ((int)(sizeof(shapes)/sizeof(shapes[0])))

/* one child's result: per-shape nanoseconds + a checksum over all Sx of all shapes */
struct result { int64_t ns[NSHAPE]; uint64_t csum; };

static void run_one_mode(int mode, struct result *out, int8_t **inbuf, int32_t **sxbuf)
{
    char ms[8]; snprintf(ms, sizeof ms, "%d", mode);
    setenv("ROCKET_BOXSUM_MODE", ms, 1);   /* first call in this child reads it */
    uint64_t csum = 0;
    for (int s = 0; s < NSHAPE; s++) {
        const struct shape *sh = &shapes[s];
        const int IHp = sh->OH + sh->KH - 1, IWp = sh->OW + sh->KW - 1;
        int8_t  *in = inbuf[s];
        int32_t *Sx = sxbuf[s];
        /* warm + iterate enough to get a stable read on small kernels */
        const int reps = 2000;
        rocket_in_window_sum_i8(in, Sx, sh->IC, IHp, IWp, sh->OH, sh->OW,
                                sh->KH, sh->KW, 1, 1, 1, 1);
        int64_t t0 = now_ns();
        for (int r = 0; r < reps; r++)
            rocket_in_window_sum_i8(in, Sx, sh->IC, IHp, IWp, sh->OH, sh->OW,
                                    sh->KH, sh->KW, 1, 1, 1, 1);
        out->ns[s] = (now_ns() - t0) / reps;
        for (int i = 0; i < sh->OH * sh->OW; i++)
            csum = csum * 1099511628211ULL + (uint64_t)(uint32_t)Sx[i];
    }
    out->csum = csum;
}

int main(void)
{
    /* allocate + fill the padded inputs once (shared by all children via fork copy-on-write) */
    int8_t  *inbuf[NSHAPE];
    int32_t *sxbuf[NSHAPE];
    srand(1234);
    for (int s = 0; s < NSHAPE; s++) {
        const struct shape *sh = &shapes[s];
        const int IHp = sh->OH + sh->KH - 1, IWp = sh->OW + sh->KW - 1;
        size_t insz = (size_t)sh->IC * IHp * IWp;
        inbuf[s] = (int8_t *)malloc(insz);
        sxbuf[s] = (int32_t *)malloc((size_t)sh->OH * sh->OW * sizeof(int32_t));
        for (size_t i = 0; i < insz; i++) inbuf[s][i] = (int8_t)((rand() % 255) - 128);
    }

    struct result res[3];
    for (int mode = 0; mode < 3; mode++) {
        int pfd[2];
        if (pipe(pfd) != 0) { perror("pipe"); return 2; }
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); return 2; }
        if (pid == 0) {
            close(pfd[0]);
            struct result r; memset(&r, 0, sizeof r);
            run_one_mode(mode, &r, inbuf, sxbuf);
            ssize_t w = write(pfd[1], &r, sizeof r);
            _exit(w == (ssize_t)sizeof r ? 0 : 3);
        }
        close(pfd[1]);
        ssize_t got = read(pfd[0], &res[mode], sizeof res[mode]);
        close(pfd[0]);
        int st = 0; waitpid(pid, &st, 0);
        if (got != (ssize_t)sizeof res[mode] || !WIFEXITED(st) || WEXITSTATUS(st) != 0) {
            fprintf(stderr, "mode %d child failed (got=%zd status=%d)\n", mode, got, st);
            return 2;
        }
    }

    const char *mname[3] = { "separable", "per-window", "scalar" };
    printf("box-sum cross-mode checksums: separable=%016llx per-window=%016llx scalar=%016llx\n",
           (unsigned long long)res[0].csum, (unsigned long long)res[1].csum,
           (unsigned long long)res[2].csum);
    int bitfail = (res[0].csum != res[2].csum) || (res[1].csum != res[2].csum);
    printf("bit-identity: %s\n", bitfail ? "FAIL" : "PASS (all three modes byte-identical)");

    printf("\n%-34s %12s %12s %12s   %8s %8s\n", "shape (sx=dx=1)",
           "separable", "per-window", "scalar", "sep/pw", "sep/scal");
    for (int s = 0; s < NSHAPE; s++) {
        double sp_pw   = res[1].ns[s] ? (double)res[1].ns[s] / (double)res[0].ns[s] : 0;
        double sp_scal = res[2].ns[s] ? (double)res[2].ns[s] / (double)res[0].ns[s] : 0;
        printf("%-34s %9lld ns %9lld ns %9lld ns   %7.2fx %7.2fx\n", shapes[s].name,
               (long long)res[0].ns[s], (long long)res[1].ns[s], (long long)res[2].ns[s],
               sp_pw, sp_scal);
    }
    (void)mname;
    for (int s = 0; s < NSHAPE; s++) { free(inbuf[s]); free(sxbuf[s]); }
    return bitfail ? 1 : 0;
}
