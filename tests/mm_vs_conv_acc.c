// SPDX-License-Identifier: GPL-3.0-or-later
// Localize the matmul-vs-conv 1x1 discrepancy: run the SAME int8 1x1 as (a) rocket_conv2d_int8
// and (b) rocket_matmul_int8_prepacked, compare both to an EXACT int64 reference C[m,oc]=sum_c
// A[m,c]*B[oc,c]. Tells us which path (if any) deviates, and whether it's nt-dependent.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_matmul.h"

static int rup(int x,int a){return (x+a-1)/a*a;}

static int test(int fd, int IH, int IW, int IC, int OC, int nt, unsigned seed)
{
    int M=IH*IW, Kp=rup(IC,32), Np=rup(OC,32);
    int8_t *A=malloc((size_t)M*IC), *B=malloc((size_t)OC*IC);
    srand(seed);
    for(size_t i=0;i<(size_t)M*IC;i++) A[i]=(int8_t)(rand()%255-127);
    for(size_t i=0;i<(size_t)OC*IC;i++) B[i]=(int8_t)(rand()%255-127);

    // exact int64 reference
    int64_t *ref=malloc((size_t)M*OC*sizeof(int64_t));
    for(int m=0;m<M;m++)for(int oc=0;oc<OC;oc++){
        int64_t s=0; for(int c=0;c<IC;c++) s+=(int)A[(size_t)m*IC+c]*(int)B[(size_t)oc*IC+c];
        ref[(size_t)m*OC+oc]=s;
    }

    // (a) conv path: in_nchw [IC][IH][IW], w_oihw [OC][IC], out_nchw [OC][IH][IW].
    //     Test BOTH the one-shot and the _mt pool (what the delegate uses).
    int8_t *in_nchw=calloc((size_t)IC*IH*IW,1), *w_oihw=malloc((size_t)OC*IC);
    int32_t *out_nchw=malloc((size_t)OC*IH*IW*sizeof(int32_t));
    int32_t *out_mt=malloc((size_t)OC*IH*IW*sizeof(int32_t));
    for(int c=0;c<IC;c++)for(int m=0;m<M;m++) in_nchw[(size_t)c*IH*IW+m]=A[(size_t)m*IC+c];
    for(int oc=0;oc<OC;oc++)for(int c=0;c<IC;c++) w_oihw[(size_t)oc*IC+c]=B[(size_t)oc*IC+c];
    rocket_conv2d_desc d={0};
    d.ic=IC;d.ih=IH;d.iw=IW;d.oc=OC;d.kh=1;d.kw=1;d.stride_y=1;d.stride_x=1;
    d.pad_top=0;d.pad_left=0;d.dil_y=1;d.dil_x=1;d.depthwise=0;
    int rc=rocket_conv2d_int8(fd,&d,in_nchw,w_oihw,out_nchw);
    rocket_conv_pool *pool=rocket_conv_pool_create(nt);
    int rcmt=pool?rocket_conv2d_int8_mt(pool,&d,in_nchw,w_oihw,out_mt):-1;
    if(pool)rocket_conv_pool_free(pool);
    long conv_bad=0,conv_max=0,mt_bad=0,mt_max=0;
    if(rc){printf("  conv rc=%d\n",rc);}else
    for(int m=0;m<M;m++)for(int oc=0;oc<OC;oc++){
        int64_t g=out_nchw[(size_t)oc*IH*IW+m], e=ref[(size_t)m*OC+oc];
        long ad=labs((long)(g-e)); if(ad){conv_bad++; if(ad>conv_max)conv_max=ad;}
    }
    if(rcmt){printf("  conv_mt rc=%d\n",rcmt);}else
    for(int m=0;m<M;m++)for(int oc=0;oc<OC;oc++){
        int64_t g=out_mt[(size_t)oc*IH*IW+m], e=ref[(size_t)m*OC+oc];
        long ad=labs((long)(g-e)); if(ad){mt_bad++; if(ad>mt_max)mt_max=ad;}
    }

    // (b) matmul path: pad B[Np][Kp], A[M][Kp]
    int8_t *Ap=calloc((size_t)M*Kp,1), *Bp=calloc((size_t)Np*Kp,1);
    for(int m=0;m<M;m++)memcpy(&Ap[(size_t)m*Kp],&A[(size_t)m*IC],IC);
    for(int oc=0;oc<OC;oc++)memcpy(&Bp[(size_t)oc*Kp],&B[(size_t)oc*IC],IC);
    int32_t *C=malloc((size_t)M*Np*sizeof(int32_t));
    rocket_i8_ctx *ctx=rocket_i8_ctx_create(nt);
    rocket_i8_weights *w=rocket_i8_weights_pack(ctx,M,Kp,Np,Bp);
    // The resident int8 path requires M%4 (Kp/Np are already %32). Small-spatial conv
    // shapes here (e.g. 5x5 -> M=25) are deliberately ineligible: they exercise the
    // CONV path, and the MM cross-check is N/A. So a pack decline on an INELIGIBLE
    // shape is a legitimate skip; a decline on an ELIGIBLE one is a real regression.
    int mm_eligible = (M%4==0);
    long mm_bad=0,mm_max=0; int mm_ran=0;
    if(!ctx||!w){
        printf(mm_eligible ? "  MM pack FAILED on an eligible shape (M%%4==0)!\n"
                           : "  MM skipped (resident int8 needs M%%4; this shape tests CONV)\n");
    }else{
        mm_ran=1;
        rocket_matmul_int8_prepacked(ctx,M,Kp,Np,Ap,C,w);
        for(int m=0;m<M;m++)for(int oc=0;oc<OC;oc++){
            int64_t g=C[(size_t)m*Np+oc], e=ref[(size_t)m*OC+oc];
            long ad=labs((long)(g-e)); if(ad){mm_bad++; if(ad>mm_max)mm_max=ad;}
        }
        rocket_i8_weights_free(ctx,w);
    }
    if(ctx)rocket_i8_ctx_free(ctx);

    // All three paths must be bit-exact vs the int64 reference: conv, conv_mt, and the
    // resident int8 matmul are each gated here (large-IC shapes included). A nonzero rc
    // (path declined) is also a failure.
    // MM fails if it ran with mismatches, OR if it was eligible but pack declined
    // (a real regression masquerading as a skip).
    int mm_fail   = (mm_ran && mm_bad) || (!mm_ran && mm_eligible);
    int conv_fail = rc || conv_bad;
    int mt_fail   = rcmt || mt_bad;
    int fail = mm_fail || conv_fail || mt_fail;
    printf("  M=%d IC=%d->Kp=%d OC=%d->Np=%d nt=%d : conv bad=%ld(max%ld) conv_mt bad=%ld(max%ld) | MM bad=%ld(max%ld)[%s] -> %s\n",
        M,IC,Kp,OC,Np,nt,conv_bad,conv_max,mt_bad,mt_max,mm_bad,mm_max,
        mm_ran ? "ran" : (mm_eligible?"PACK-FAIL":"skip"),
        fail ? (mm_fail?"MM-FAIL":conv_fail?"CONV-FAIL":"CONVMT-FAIL") : "OK");
    free(A);free(B);free(ref);free(in_nchw);free(w_oihw);free(out_nchw);free(out_mt);free(Ap);free(Bp);free(C);
    return fail?1:0;
}

int main(void){
    int fd=rocket_open();
    if(fd<0){printf("no NPU (%d)\n",fd);return 2;}
    int fail=0;
    // aligned (no pad): the heavy MobileDet 1x1s
    fail|=test(fd,20,20, 96,768, 4, 1);
    fail|=test(fd,20,20,768, 96, 4, 2);
    fail|=test(fd,10,10,960,384, 4, 3);
    // padded K and/or N (the MobileNet project / SSD predictor shapes)
    fail|=test(fd,20,20, 40,320, 4, 4);   // K pad 40->64
    fail|=test(fd,10,10,960,120, 4, 5);   // N pad 120->128
    fail|=test(fd,10,10,384,546, 4, 6);   // N pad 546->576
    fail|=test(fd,80,80,128, 16, 4, 7);   // N pad 16->32, big M
    // nt sweep on one shape (does the matmul depend on worker count?)
    fail|=test(fd,20,20, 96,768, 1, 1);
    fail|=test(fd,20,20, 96,768, 2, 1);
    fail|=test(fd,20,20, 40,320, 1, 4);
    fail|=test(fd,20,20, 40,320, 2, 4);
    // characterize the CONV bug: sweep spatial M at the failing IC=768 OC=96
    printf("-- conv-bug M sweep @ IC=768 OC=96 --\n");
    fail|=test(fd, 5, 5,768, 96, 4, 8);
    fail|=test(fd,10,10,768, 96, 4, 8);
    fail|=test(fd,14,14,768, 96, 4, 8);
    fail|=test(fd,20,20,768, 96, 4, 8);
    rocket_close(fd);
    printf("==== %s ====\n", fail?"FAIL":"PASS");
    return fail?1:0;
}
