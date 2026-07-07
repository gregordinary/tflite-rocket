// SPDX-License-Identifier: GPL-3.0-or-later
// Does the fp16 matmul resonate at the SAME (Mtile,Ktile) the int8 path corrupts?
// fp16 uses a C2=8, 2-byte feature cube; int8 uses C2=16, 1-byte. If fp16 is CLEAN at
// M=144,K=672 (where int8 corrupts the last rows), the bug is int8-cube-specific.
// Inputs are {-1,0,1} so the exact integer dot-product (|sum|<=K<2048) is fp16-exact;
// any row off by >> rounding is the resonance bug.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rocket_npu.h"
#include "rocket_matmul.h"

static int test(int fd,int M,int K,int N,unsigned seed){
    _Float16 *A=malloc((size_t)M*K*sizeof(_Float16)),*B=malloc((size_t)N*K*sizeof(_Float16));
    _Float16 *C=malloc((size_t)M*N*sizeof(_Float16));
    srand(seed);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=(_Float16)(rand()%3-1);
    for(size_t i=0;i<(size_t)N*K;i++)B[i]=(_Float16)(rand()%3-1);
    long *ref=malloc((size_t)M*N*sizeof(long));
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){long s=0;
        for(int c=0;c<K;c++)s+=(long)(float)A[(size_t)m*K+c]*(long)(float)B[(size_t)n*K+c];
        ref[(size_t)m*N+n]=s;}
    int rc=rocket_matmul_fp16(fd,M,K,N,A,B,C);
    long bad=0; int rmin=1<<30,rmax=-1; double maxe=0;
    if(rc){printf("M=%d K=%d N=%d: rc=%d\n",M,K,N,rc);}
    else{
      for(int m=0;m<M;m++)for(int n=0;n<N;n++){
        double e=(double)C[(size_t)m*N+n]-(double)ref[(size_t)m*N+n];
        if(e<0)e=-e; if(e>0.5){bad++; if(m<rmin)rmin=m; if(m>rmax)rmax=m; if(e>maxe)maxe=e;}
      }
      int Mt,Kt,Nt; rocket_matmul_plan(M,K,N,&Mt,&Kt,&Nt);
      printf("M=%d K=%d N=%d: Mt=%d Kt=%d Nt=%d  bad=%ld maxe=%.0f%s%s\n",M,K,N,Mt,Kt,Nt,bad,maxe,
        bad?" rows[":"",bad?"":"");
      if(bad)printf("   rows[%d..%d] of %d -> %s\n",rmin,rmax,M, "FP16 RESONATES");
    }
    free(A);free(B);free(C);free(ref);
    return bad?1:0;
}
int main(void){
    int fd=rocket_open(); if(fd<0){printf("no NPU\n");return 2;}
    int f=0;
    f|=test(fd,144,672,128,1);  // int8 corrupts last 5 rows here
    f|=test(fd,192,672,128,2);  // int8 corrupts last 6
    f|=test(fd,240,672,128,3);  // int8 corrupts last 8
    f|=test(fd,144,224,128,4);  // int8 g7 bad
    f|=test(fd,256,672,128,5);  // int8 GOOD Mtile (control)
    f|=test(fd,100,672,128,6);  // int8 GOOD Mtile (control)
    rocket_close(fd);
    printf("==== %s ====\n",f?"SOME FP16 RESONANCE":"FP16 ALL CLEAN");
    // fp16 is expected to be CLEAN (the resonance bug is int8-cube-specific). Exit
    // nonzero if fp16 ever resonates, so this stays a real regression gate.
    return f?1:0;
}
