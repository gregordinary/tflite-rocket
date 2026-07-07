// SPDX-License-Identifier: GPL-3.0-or-later
// Conv int8 CBUF feature-bank-slack resonance sweep — the conv analog of mm_nt_det.c.
//
// AUDIT: gen_conv2d_int8_fill (npu_regcmd.c) sets data_bank = ceil(feat_bytes/32KB),
// exact-fit — the same sizing that made the int8 MATMUL feature-cube DMA over-read one
// CBUF bank at near-bank-full (Mtile,Ktile) geometries, garbling a tile's tail output
// rows. The conv feature cube is the
// same C2=16, 1-byte cube, so it is structurally susceptible. This sweep runs a
// SINGLE-JOB int8 conv (feature <= 8 banks, weight <= 4 banks, IC%32 => the one-job
// fast path in conv2d_int8_run, so the GENERATOR's data_bank is exercised directly, not
// the host tiler decomposition) across a grid of (IH, IW, IC) for 1x1 and KxK, vs an
// exact int64 reference. It flags any shape whose error concentrates in the last output
// rows (the over-read signature: ~1e5-1e6, all columns of the last few rows).
//
// A/B the +1 feature slack on HW WITHOUT a recompile (driver sentinel):
//   sudo ROCKET_CONV_I8_FDBANK_EXTRA=0 ./conv_bank_slack   # exact-fit  (current)
//   sudo ROCKET_CONV_I8_FDBANK_EXTRA=1 ./conv_bank_slack   # +1 slack   (the candidate fix)
// extra=-1 (under-allocate) should make any resonant shape far worse.
//
// Driver-level note: the uint8 (Option D) path host-recenters to signed int8 and calls
// this SAME rocket_conv2d_int8 — the over-read is a DMA GEOMETRY effect, independent of
// byte values — so a signed-int8 geometry sweep fully covers uint8 at the driver. UINT8=1
// just fills bytes from the recentered range to make that explicit. Depthwise shares the
// identical feature-bank computation (gen_conv2d_int8_fill lines for fd_bytes/data_bank
// run for depthwise=0 and =1 alike); DW=1 exercises it through the DW runtime where it
// routes.
//
// Env: IC_LIST (csv, %32), IW, OC, KH, KW, IH_MIN, IH_MAX, IH_STEP, SEED, UINT8, VERBOSE.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rocket_npu.h"
#include "rocket_conv.h"

#define CBUF_BANK 32768

static int geti(const char *k, int dflt) { const char *e = getenv(k); return e ? atoi(e) : dflt; }

// One shape: single-job int8 1x1/KxK conv vs exact int64 ref. Returns 1 if any error.
static int test(int fd, int IH, int IW, int IC, int OC, int KH, int KW,
                int uint8_mode, int verbose, unsigned seed, long *worst_max)
{
    // SAME-pad so OH/OW == IH/IW (keeps the spatial map square; KxK feature includes halo).
    int sy = 1, sx = 1, dy = 1, dx = 1;
    int pt = (KH - 1) / 2, pl = (KW - 1) / 2;
    int OH = IH, OW = IW;
    size_t feat_bytes = (size_t)IH * IW * IC;       // single-job feature cube (1 B/elem)
    size_t wt_bytes   = (size_t)OC * IC * KH * KW;  // direct weight cube
    int fd_banks = (int)((feat_bytes + CBUF_BANK - 1) / CBUF_BANK);
    double fill  = 100.0 * (double)feat_bytes / ((double)fd_banks * CBUF_BANK);

    // keep it on the one-job fast path so the generator's data_bank is what we test
    if (feat_bytes > (size_t)8 * CBUF_BANK) return 0;
    if (wt_bytes   > (size_t)4 * CBUF_BANK) return 0;

    int M = IH * IW;
    int8_t *A = malloc((size_t)M * IC), *B = malloc(wt_bytes);
    srand(seed);
    int lo = uint8_mode ? -128 : -127, hi = uint8_mode ? 127 : 127;
    for (size_t i = 0; i < (size_t)M * IC; i++) A[i] = (int8_t)(rand() % (hi - lo + 1) + lo);
    for (size_t i = 0; i < wt_bytes; i++)       B[i] = (int8_t)(rand() % (hi - lo + 1) + lo);

    // in_nchw [IC][IH][IW], w_oihw [OC][IC][KH][KW], out_nchw [OC][OH][OW]
    int8_t  *in_nchw  = malloc((size_t)IC * IH * IW);
    int8_t  *w_oihw   = malloc(wt_bytes);
    int32_t *out_nchw = malloc((size_t)OC * OH * OW * sizeof(int32_t));
    for (int c = 0; c < IC; c++) for (int m = 0; m < M; m++)
        in_nchw[(size_t)c * IH * IW + m] = A[(size_t)m * IC + c];
    memcpy(w_oihw, B, wt_bytes);

    rocket_conv2d_desc d = {0};
    d.ic = IC; d.ih = IH; d.iw = IW; d.oc = OC; d.kh = KH; d.kw = KW;
    d.stride_y = sy; d.stride_x = sx; d.pad_top = pt; d.pad_left = pl;
    d.dil_y = dy; d.dil_x = dx; d.depthwise = 0;
    int rc = rocket_conv2d_int8(fd, &d, in_nchw, w_oihw, out_nchw);

    long bad = 0, mx = 0; int bad_row_lo = OH, bad_row_hi = -1;
    if (rc == 0) {
        for (int oc = 0; oc < OC; oc++)
        for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++) {
            // exact int64 ref for this output pixel (SAME pad, zero outside)
            int64_t e = 0;
            for (int c = 0; c < IC; c++)
            for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++) {
                int ih = oh * sy - pt + kh * dy, iw = ow * sx - pl + kw * dx;
                if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) continue;
                e += (int)A[((size_t)(ih * IW + iw)) * IC + c] *
                     (int)B[(((size_t)oc * IC + c) * KH + kh) * KW + kw];
            }
            int64_t g = out_nchw[((size_t)oc * OH + oh) * OW + ow];
            long ad = labs((long)(g - e));
            if (ad) { bad++; if (ad > mx) mx = ad; if (oh < bad_row_lo) bad_row_lo = oh; if (oh > bad_row_hi) bad_row_hi = oh; }
        }
    }

    int tail = (bad && bad_row_lo >= OH - 4);   // error confined to the last <=4 output rows
    if (bad || verbose || fill >= 95.0)
        printf("  IH=%3d IW=%2d IC=%4d OC=%3d %dx%d  feat=%zuB=%d bank(%.1f%%)  rc=%d  bad=%ld max=%ld%s%s\n",
               IH, IW, IC, OC, KH, KW, feat_bytes, fd_banks, fill, rc, bad, mx,
               bad ? (tail ? "  [TAIL-ROW rows " : "  [bad rows ") : "",
               bad ? "]" : "");
    if (bad && (bad || tail)) printf("        bad output rows %d..%d of %d\n", bad_row_lo, bad_row_hi, OH);
    if (mx > *worst_max) *worst_max = mx;

    free(A); free(B); free(in_nchw); free(w_oihw); free(out_nchw);
    return bad ? 1 : 0;
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) { printf("no NPU (%d)\n", fd); return 2; }

    /* DEFAULTS = the decisive regression gate: IW=1 makes the conv feature DMA identical
     * to gen_matmul_int8's (datain_width=1), so the matmul's near-bank-full resonance is
     * reachable here. With the fix this PASSES; reverting it (ROCKET_CONV_I8_FDBANK_EXTRA=-1)
     * FAILS. Env vars override for broad exploration (IW=2/4, KxK, UINT8, other IC/IH). */
    int IW      = geti("IW", 1);
    int OC      = geti("OC", 32);
    int KH      = geti("KH", 1), KW = geti("KW", 1);
    int IH_MIN  = geti("IH_MIN", 8), IH_MAX = geti("IH_MAX", 256), IH_STEP = geti("IH_STEP", 1);
    int uint8_m = geti("UINT8", 0);
    int verbose = geti("VERBOSE", 0);
    unsigned seed = (unsigned)geti("SEED", 12345);

    // default IC list = the matmul's "hot" K-groups (17,21,25,29,33,37)*32 — the channel
    // counts whose near-full feature cube resonates at IW=1. 672 (=21*32) is the classic one.
    int ic_def[] = { 544, 672, 800, 928, 1056, 1184 };
    int ic_list[64]; int n_ic = 0;
    const char *icl = getenv("IC_LIST");
    if (icl) {
        char buf[512]; strncpy(buf, icl, sizeof buf - 1); buf[sizeof buf - 1] = 0;
        for (char *t = strtok(buf, ","); t && n_ic < 64; t = strtok(NULL, ",")) ic_list[n_ic++] = atoi(t);
    } else { for (size_t i = 0; i < sizeof ic_def / sizeof *ic_def; i++) ic_list[n_ic++] = ic_def[i]; }

    const char *ex = getenv("ROCKET_CONV_I8_FDBANK_EXTRA");
    printf("conv int8 bank-slack sweep: IW=%d OC=%d %dx%d IH=[%d..%d/%d] UINT8=%d FDBANK_EXTRA=%s\n",
           IW, OC, KH, KW, IH_MIN, IH_MAX, IH_STEP, uint8_m, ex ? ex : "(unset=exact-fit)");

    int fail = 0; long worst = 0; int shapes = 0;
    for (int i = 0; i < n_ic; i++) {
        for (int IH = IH_MIN; IH <= IH_MAX; IH += IH_STEP) {
            fail |= test(fd, IH, IW, ic_list[i], OC, KH, KW, uint8_m, verbose, seed + IH * 131 + i, &worst);
            shapes++;
        }
    }
    rocket_close(fd);
    printf("==== %s ====  (%d shapes, worst max-abs err = %ld)\n", fail ? "FAIL" : "PASS", shapes, worst);
    return fail ? 1 : 0;
}
