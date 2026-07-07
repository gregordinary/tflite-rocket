// SPDX-License-Identifier: GPL-3.0-or-later
// Bisect the int8-conv large-feature-cube bug. Probe SINGLE-JOB 1x1 int8 convs
// (feature <= budget so rocket_conv2d_int8 fast-paths == one CNA pass, no tiling)
// across (IC, IH, IW) to find the exact breaking threshold + which axis triggers
// it, then confirm 3x3 and uint8 (recenter trick) reproduce. Compares each to the
// exact int64 reference C[oc,h,w] = sum_ic A[ic,h,w]*B[oc,ic].
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rocket_npu.h"
#include "rocket_conv.h"

#define BANK 32768

/* one direct int8 conv, in_nchw[IC][IH][IW], w_oihw[OC][IC][KH][KW], out_nchw[OC][OH][OW].
 * Returns bad-elem count (0 == bit-exact), -1 on rc!=0. Prints a one-line verdict. */
static long probe(int fd, int IC, int IH, int IW, int OC, int KH, int KW, unsigned seed)
{
    int OH = IH - KH + 1, OW = IW - KW + 1;
    if (OH <= 0 || OW <= 0) return 0;
    size_t insz = (size_t)IC*IH*IW, wsz = (size_t)OC*IC*KH*KW, osz = (size_t)OC*OH*OW;
    int8_t *in = malloc(insz), *w = malloc(wsz);
    int32_t *out = malloc(osz*sizeof(int32_t));
    srand(seed);
    for (size_t i=0;i<insz;i++) in[i]=(int8_t)(rand()%255-127);
    for (size_t i=0;i<wsz;i++)  w[i]=(int8_t)(rand()%255-127);

    rocket_conv2d_desc d={0};
    d.ic=IC;d.ih=IH;d.iw=IW;d.oc=OC;d.kh=KH;d.kw=KW;d.stride_y=1;d.stride_x=1;
    d.pad_top=0;d.pad_left=0;d.dil_y=1;d.dil_x=1;d.depthwise=0;
    int rc = rocket_conv2d_int8(fd,&d,in,w,out);

    long bad=0,mx=0;
    if (rc) { bad=-1; }
    else for (int oc=0;oc<OC;oc++) for (int oh=0;oh<OH;oh++) for (int ow=0;ow<OW;ow++) {
        int64_t s=0;
        for (int ic=0;ic<IC;ic++) for (int kh=0;kh<KH;kh++) for (int kw=0;kw<KW;kw++)
            s += (int)in[((size_t)ic*IH+oh+kh)*IW+ow+kw] *
                 (int)w[(((size_t)oc*IC+ic)*KH+kh)*KW+kw];
        int64_t g = out[((size_t)oc*OH+oh)*OW+ow];
        long ad = labs((long)(g-s)); if(ad){bad++; if(ad>mx)mx=ad;}
    }
    size_t fb = (insz + BANK - 1)/BANK;
    printf("  IC=%4d IH=%2d IW=%2d OC=%d %dx%d : feat=%7zuB fd_banks=%2zu : %s (bad=%ld max=%ld)\n",
        IC,IH,IW,OC,KH,KW,insz,fb, bad==0?"OK ":(bad<0?"RC ":"BAD"), bad, mx);
    free(in);free(w);free(out);
    return bad;
}

int main(void){
    int fd=rocket_open();
    if(fd<0){printf("no NPU (%d)\n",fd);return 2;}
    // Accumulate EVERY probe (probe() returns 0=bit-exact, >0=bad elems, -1=rc!=0;
    // !=0 is a failure). A regression gate over the int8-conv large-feature-cube tiling
    // (rocket_conv.c large-IC tiling + bank-slack): every probe below — across the full
    // tile-geometry sweep — must be bit-exact.
    long fail=0;
    #define P(...) do { fail += (probe(fd, __VA_ARGS__) != 0); } while(0)

    printf("== IC=768 OC=96 1x1 single-job feature/bank sweep (all <= 256KB budget) ==\n");
    // square + rectangular; feature grows -> fd_banks 4..8. Find the break.
    P(768,14,14,96,1,1,11);   // 150528  5 banks (known good)
    P(768,12,16,96,1,1,11);               // 147456  5
    P(768,16,12,96,1,1,11);               // 147456  5
    P(768,12,20,96,1,1,11);               // 184320  6
    P(768,20,12,96,1,1,11);               // 184320  6
    P(768,14,20,96,1,1,11);               // 215040  7
    P(768,20,14,96,1,1,11);               // 215040  7
    P(768,16,16,96,1,1,11);               // 196608  6
    P(768,16,20,96,1,1,11);               // 245760  8
    P(768,20,16,96,1,1,11);               // 245760  8
    P(768,17,20,96,1,1,11);               // 261120  8  (was the failing tile, as single job)
    P(768,20,17,96,1,1,11);               // 261120  8

    printf("== IC sweep at the same ~8-bank feature (vary IC, hold feat near budget) ==\n");
    P(256,20,20,96,1,1,12);               // 102400  4
    P(512,16,16,96,1,1,12);               // 131072  4
    P(512,16,20,96,1,1,12);               // 163840  5
    P(512,20,20,96,1,1,12);               // 204800  7
    P(512,20,24,96,1,1,12);               // 245760  8
    P(960,12,16,96,1,1,12);               // 184320  6
    P(960,14,16,96,1,1,12);               // 215040  7
    P(960,16,16,96,1,1,12);               // 245760  8

    printf("== 3x3 large-IC single job (KxK, not 1x1) ==\n");
    P(256,18,18,64,3,3,13);               // 256*18*18=82944 -> OH=16
    P(512,18,18,64,3,3,13);               // 165888
    P(768,18,18,64,3,3,13);               // 248832  8

    printf("== short-height 1x1 single job at large IC (the rht remainder tile) ==\n");
    // M=400 1x1 tiles into rht=17 + a 3-row remainder; bad=5760=3*20*96 == this tile.
    for (int ih=2; ih<=8; ih++) P(768,ih,20,96,1,1,14);
    // does the remainder bug depend on IC? short height, vary IC
    P(256,3,20,96,1,1,15);
    P(512,3,20,96,1,1,15);
    P(768,3,20,96,1,1,15);

    printf("== reproduce the ORIGINAL tiled failures (force tiling, >budget) ==\n");
    P(768,20,20,96,1,1,16);               // 307200 -> tiles rht=17 + rh=3 (was BAD)
    P(960,20,20,384,1,1,16);              // big

    #undef P
    rocket_close(fd);
    printf("==== %s ====\n", fail?"SOME PROBES FAILED":"ALL PROBES OK");
    return fail?1:0;
}
