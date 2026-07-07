// SPDX-License-Identifier: GPL-3.0-or-later
// mm_nt_det.c — does the resident int8 matmul depend on worker count (nt)?
// For each (M,IC,OC) routed 1x1 shape: pad K/N to %32, run rocket_matmul_int8_prepacked
// at nt=1,2,3,4, compare EVERY result to the exact int64 reference AND to the nt=1 result.
// A shape where nt!=1 disagrees with nt=1 (or with ref) is THE nondeterminism. Matmul-only
// (no conv) so it iterates fast. Shapes = EfficientDet-Lite0 routed 1x1s + MobileDet heavy ones.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rocket_npu.h"
#include "rocket_matmul.h"

static int rup(int x,int a){return (x+a-1)/a*a;}

// run one (M,Kp,Np) through the matmul at worker count nt; fill out[M*Np] int32.
static int run_mm(int M,int Kp,int Np,const int8_t*Ap,const int8_t*Bp,int32_t*out,int nt)
{
    rocket_i8_ctx *ctx=rocket_i8_ctx_create(nt);
    if(!ctx) return -1;
    rocket_i8_weights *w=rocket_i8_weights_pack(ctx,M,Kp,Np,Bp);
    if(!w){ rocket_i8_ctx_free(ctx); return -2; }
    int rc=rocket_matmul_int8_prepacked(ctx,M,Kp,Np,Ap,out,w);
    rocket_i8_weights_free(ctx,w);
    rocket_i8_ctx_free(ctx);
    return rc;
}

static int test(int M,int IC,int OC,unsigned seed)
{
    int Kp=rup(IC,32), Np=rup(OC,32);
    int8_t *A=malloc((size_t)M*IC), *B=malloc((size_t)OC*IC);
    srand(seed);
    for(size_t i=0;i<(size_t)M*IC;i++) A[i]=(int8_t)(rand()%255-127);
    for(size_t i=0;i<(size_t)OC*IC;i++) B[i]=(int8_t)(rand()%255-127);

    int64_t *ref=malloc((size_t)M*OC*sizeof(int64_t));
    for(int m=0;m<M;m++)for(int oc=0;oc<OC;oc++){
        int64_t s=0; for(int c=0;c<IC;c++) s+=(int)A[(size_t)m*IC+c]*(int)B[(size_t)oc*IC+c];
        ref[(size_t)m*OC+oc]=s;
    }
    int8_t *Ap=calloc((size_t)M*Kp,1), *Bp=calloc((size_t)Np*Kp,1);
    for(int m=0;m<M;m++)memcpy(&Ap[(size_t)m*Kp],&A[(size_t)m*IC],IC);
    for(int oc=0;oc<OC;oc++)memcpy(&Bp[(size_t)oc*Kp],&B[(size_t)oc*IC],IC);

    int32_t *o1=malloc((size_t)M*Np*sizeof(int32_t));
    int nts[]={1,2,3,4};
    int fail=0;
    printf("M=%-6d IC=%-4d->Kp=%-4d OC=%-4d->Np=%-4d :",M,IC,Kp,OC,Np);
    int32_t *base=NULL;
    for(int i=0;i<4;i++){
        int nt=nts[i];
        int32_t *o=(i==0)?o1:malloc((size_t)M*Np*sizeof(int32_t));
        int rc=run_mm(M,Kp,Np,Ap,Bp,o,nt);
        if(rc){ printf(" nt%d:rc%d",nt,rc); fail=1; if(o!=o1)free(o); continue; }
        // vs ref (only over real OC cols)
        long vref=0; for(int m=0;m<M;m++)for(int oc=0;oc<OC;oc++){
            if(o[(size_t)m*Np+oc]!=ref[(size_t)m*OC+oc])vref++;
        }
        // vs nt=1 (full padded width)
        long vbase=0; if(i>0) for(size_t j=0;j<(size_t)M*Np;j++) if(o[j]!=base[j])vbase++;
        printf(" nt%d[ref%ld%s]",nt,vref,(i>0)?(vbase?" DIFF":" =1"):"");
        if(vref||(i>0&&vbase))fail=1;
        if(i==0)base=o1;
        else free(o);
    }
    printf(" -> %s\n",fail?"FAIL":"ok");
    free(A);free(B);free(ref);free(Ap);free(Bp);free(o1);
    return fail;
}

// deep diagnostic for ONE shape at nt=1: resolved tiling + mismatch map (rows/cols/maxerr).
// Returns the bad-element count (0 == bit-exact) so callers can gate on it.
static long diag(int M,int IC,int OC,unsigned seed)
{
    int Kp=rup(IC,32), Np=rup(OC,32);
    int Mt,Kt,Nt; rocket_matmul_plan_int8(M,Kp,Np,&Mt,&Kt,&Nt);
    printf("DIAG M=%d IC=%d(Kp=%d) OC=%d(Np=%d): Mt=%d Kt=%d Nt=%d  nMt=%d nNt=%d nKt=%d\n",
        M,IC,Kp,OC,Np,Mt,Kt,Nt,(M+Mt-1)/Mt,(Np+Nt-1)/Nt,(Kp+Kt-1)/Kt);
    int8_t *A=malloc((size_t)M*IC), *B=malloc((size_t)OC*IC);
    srand(seed);
    for(size_t i=0;i<(size_t)M*IC;i++) A[i]=(int8_t)(rand()%255-127);
    for(size_t i=0;i<(size_t)OC*IC;i++) B[i]=(int8_t)(rand()%255-127);
    int64_t *ref=malloc((size_t)M*OC*sizeof(int64_t));
    for(int m=0;m<M;m++)for(int oc=0;oc<OC;oc++){int64_t s=0;
        for(int c=0;c<IC;c++)s+=(int)A[(size_t)m*IC+c]*(int)B[(size_t)oc*IC+c];
        ref[(size_t)m*OC+oc]=s;}
    int8_t *Ap=calloc((size_t)M*Kp,1),*Bp=calloc((size_t)Np*Kp,1);
    for(int m=0;m<M;m++)memcpy(&Ap[(size_t)m*Kp],&A[(size_t)m*IC],IC);
    for(int oc=0;oc<OC;oc++)memcpy(&Bp[(size_t)oc*Kp],&B[(size_t)oc*IC],IC);
    int32_t *o=malloc((size_t)M*Np*sizeof(int32_t));
    run_mm(M,Kp,Np,Ap,Bp,o,1);
    long bad=0,maxe=0; int rmin=1<<30,rmax=-1,cmin=1<<30,cmax=-1;
    for(int m=0;m<M;m++)for(int oc=0;oc<OC;oc++){
        long e=labs((long)(o[(size_t)m*Np+oc]-ref[(size_t)m*OC+oc]));
        if(e){bad++; if(e>maxe)maxe=e; if(m<rmin)rmin=m; if(m>rmax)rmax=m;
              if(oc<cmin)cmin=oc; if(oc>cmax)cmax=oc;}
    }
    printf("   bad=%ld maxerr=%ld rows[%d..%d] cols[%d..%d]\n",bad,maxe,rmin,rmax,cmin,cmax);
    // per-row wrong-col count for first/last few rows around the boundary
    if(bad>0 && bad<=24*OC)
      for(int m=0;m<M;m++){int w=0;for(int oc=0;oc<OC;oc++) if(o[(size_t)m*Np+oc]!=ref[(size_t)m*OC+oc])w++;
        if(w) printf("     row %d: %d wrong cols\n",m,w);}
    free(A);free(B);free(ref);free(Ap);free(Bp);free(o);
    return bad;
}

int main(void){
    int fd=rocket_open();
    if(fd<0){printf("no NPU (%d)\n",fd);return 2;}
    rocket_close(fd);
    if(getenv("DIAG")){ return diag(400,672,112,26) ? 1 : 0; }
    if(getenv("MSWEEP")){
        // single-tile (M<=256) Mtile sweep at K=672, N=128: which Mtile are bad?
        long sweepbad=0;
        for(int M=128;M<=256;M+=4){
            int Kp=672,Np=128;
            int8_t*A=malloc((size_t)M*Kp),*B=malloc((size_t)Np*Kp);
            srand(M); for(size_t i=0;i<(size_t)M*Kp;i++)A[i]=(int8_t)(rand()%255-127);
            for(size_t i=0;i<(size_t)Np*Kp;i++)B[i]=(int8_t)(rand()%255-127);
            int64_t*ref=malloc((size_t)M*Np*sizeof(int64_t));
            for(int m=0;m<M;m++)for(int n=0;n<Np;n++){int64_t s=0;
                for(int c=0;c<Kp;c++)s+=(int)A[(size_t)m*Kp+c]*(int)B[(size_t)n*Kp+c];
                ref[(size_t)m*Np+n]=s;}
            int32_t*o=malloc((size_t)M*Np*sizeof(int32_t));
            run_mm(M,Kp,Np,A,B,o,1);
            long bad=0;int rmin=1<<30,rmax=-1;
            for(int m=0;m<M;m++)for(int n=0;n<Np;n++) if(o[(size_t)m*Np+n]!=ref[(size_t)m*Np+n]){bad++;if(m<rmin)rmin=m;if(m>rmax)rmax=m;}
            printf("Mtile=%-3d K=672 N=128: bad=%-6ld %s\n",M,bad,bad?"FAIL rows":"" );
            if(bad)printf("   rows[%d..%d] of %d\n",rmin,rmax,M);
            sweepbad+=bad;
            free(A);free(B);free(ref);free(o);
        }
        return sweepbad?1:0;
    }
    if(getenv("KSWEEP")){
        // Mtile=144 fixed, sweep K (multiple of 32) to map bad K set
        long sweepbad=0;
        for(int K=64;K<=2048;K+=32){
            int M=144,Np=128;
            int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)Np*K);
            srand(K); for(size_t i=0;i<(size_t)M*K;i++)A[i]=(int8_t)(rand()%255-127);
            for(size_t i=0;i<(size_t)Np*K;i++)B[i]=(int8_t)(rand()%255-127);
            int64_t*ref=malloc((size_t)M*Np*sizeof(int64_t));
            for(int m=0;m<M;m++)for(int n=0;n<Np;n++){int64_t s=0;
                for(int c=0;c<K;c++)s+=(int)A[(size_t)m*K+c]*(int)B[(size_t)n*K+c];
                ref[(size_t)m*Np+n]=s;}
            int32_t*o=malloc((size_t)M*Np*sizeof(int32_t));
            run_mm(M,K,Np,A,B,o,1);
            long bad=0; for(int m=0;m<M;m++)for(int n=0;n<Np;n++) if(o[(size_t)m*Np+n]!=ref[(size_t)m*Np+n])bad++;
            if(bad)printf("Mtile=144 K=%-4d N=128: bad=%ld FAIL\n",K,bad);
            sweepbad+=bad;
            free(A);free(B);free(ref);free(o);
        }
        printf("(KSWEEP done; only FAILs printed)\n");
        return sweepbad?1:0;
    }
    if(getenv("MAP2D")){
        // Full Mtile x K-group bad-set map (nt=1, single tile via M<=256). N small for fast ref.
        // For each Mtile (4..256 step4) print the list of bad K-groups. Reveals the resonance law.
        int N=32;
        long mapbad=0;
        printf("Mtile : bad K-groups (K=g*32), N=%d\n",N);
        for(int M=4;M<=256;M+=4){
            int8_t*B=malloc((size_t)N*40*32);
            srand(M);
            char line[1024]; int li=0; int any=0;
            for(int g=1;g<=40;g++){
                int K=g*32;
                int Mt,Kt,Nt; if(rocket_matmul_plan_int8(M,K,N,&Mt,&Kt,&Nt)<0) continue;
                if(Mt!=M||Kt!=K) continue;   // only pure single-tile cases (isolate geometry)
                int8_t*A=malloc((size_t)M*K);
                for(size_t i=0;i<(size_t)M*K;i++)A[i]=(int8_t)(rand()%255-127);
                for(size_t i=0;i<(size_t)N*K;i++)B[i]=(int8_t)(rand()%255-127);
                int64_t*ref=malloc((size_t)M*N*sizeof(int64_t));
                for(int m=0;m<M;m++)for(int n=0;n<N;n++){int64_t s=0;
                    for(int c=0;c<K;c++)s+=(int)A[(size_t)m*K+c]*(int)B[(size_t)n*K+c];
                    ref[(size_t)m*N+n]=s;}
                int32_t*o=malloc((size_t)M*N*sizeof(int32_t));
                run_mm(M,K,N,A,B,o,1);
                long bad=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++) if(o[(size_t)m*N+n]!=ref[(size_t)m*N+n])bad++;
                if(bad){ li+=snprintf(line+li,sizeof(line)-li,"%d ",g); any=1; mapbad+=bad; }
                free(A);free(ref);free(o);
            }
            free(B);
            if(any) printf("M=%3d : %s\n",M,line);
        }
        return mapbad?1:0;
    }
    if(getenv("GRID")){
        // Comprehensive M x K x N sweep on the UNGUARDED matmul. For each failure print
        // resolved Kt and the per-K-tile group sizes, to confirm the complete bad condition.
        int Ms[]={144,192,240,400,100,1600};
        int Ns[]={96,128,160,320};
        long total=0,fails=0;
        for(int kg=1;kg<=48;kg++){
            int K=kg*32;
            for(unsigned mi=0;mi<sizeof(Ms)/sizeof(Ms[0]);mi++){
              for(unsigned ni=0;ni<sizeof(Ns)/sizeof(Ns[0]);ni++){
                int M=Ms[mi],N=Ns[ni];
                int Mt,Kt,Nt; if(rocket_matmul_plan_int8(M,K,N,&Mt,&Kt,&Nt)<0) continue;
                int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)N*K);
                srand(kg*1000+mi*10+ni);
                for(size_t i=0;i<(size_t)M*K;i++)A[i]=(int8_t)(rand()%255-127);
                for(size_t i=0;i<(size_t)N*K;i++)B[i]=(int8_t)(rand()%255-127);
                int64_t*ref=malloc((size_t)M*N*sizeof(int64_t));
                for(int m=0;m<M;m++)for(int n=0;n<N;n++){int64_t s=0;
                    for(int c=0;c<K;c++)s+=(int)A[(size_t)m*K+c]*(int)B[(size_t)n*K+c];
                    ref[(size_t)m*N+n]=s;}
                int32_t*o=malloc((size_t)M*N*sizeof(int32_t));
                run_mm(M,K,N,A,B,o,1);
                long bad=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++) if(o[(size_t)m*N+n]!=ref[(size_t)m*N+n])bad++;
                total++;
                if(bad){
                    fails++;
                    // per-tile group sizes (uniform Kt + remainder)
                    int nKt=(K+Kt-1)/Kt, rem=K-(nKt-1)*Kt;
                    int badtile=0; if(((Kt/32)%14)==7)badtile=1; if(((rem/32)%14)==7)badtile=1;
                    printf("FAIL M=%-4d K=%-4d(g%d) N=%-3d : Mt=%d Kt=%d(g%d) nKt=%d rem=%d(g%d) bad=%-5ld  predicted=%s\n",
                        M,K,kg,N,Mt,Kt,Kt/32,nKt,rem,rem/32,bad,badtile?"YES":"NO-MISS!");
                }
                free(A);free(B);free(ref);free(o);
              }
            }
        }
        printf("GRID: %ld cases, %ld FAIL\n",total,fails);
        return fails?1:0;
    }
    if(getenv("SWEEP")){
        int shp[][3]={ {144,672,112},{148,672,112},{160,672,112},{256,672,112},{400,672,112},
                       {400,672,80},{400,672,96},{400,672,160},{400,672,224},
                       {400,640,112},{400,704,112},{400,768,112},{400,608,112},
                       {400,672,128},{400,672,256} };
        long sweepbad=0;
        for(unsigned i=0;i<sizeof(shp)/sizeof(shp[0]);i++) sweepbad+=diag(shp[i][0],shp[i][1],shp[i][2],26+i);
        return sweepbad?1:0;
    }
    int fail=0;
    // EfficientDet-Lite0 routed 1x1 shapes (M%4||1; K,N pad to %32), big-K first.
    fail|=test(100,1152,320,11);
    fail|=test(100,192,1152,12);
    fail|=test(100,1152,192,13);
    fail|=test(100,672,192,14);
    fail|=test(100,320,64,15);
    fail|=test(1600,64,64,16);
    fail|=test(400,64,64,17);
    fail|=test(100,64,64,18);
    // padded (K and/or N not %32) routed shapes + the huge-M ones
    fail|=test(1600,64,810,21);
    fail|=test(100,64,810,22);
    fail|=test(25600,16,96,23);
    fail|=test(6400,24,144,24);
    fail|=test(400,112,672,25);
    fail|=test(400,672,112,26);
    fail|=test(400,480,80,27);
    // MobileDet heavy ones for cross-check
    fail|=test(400,768,96,31);
    fail|=test(400,96,768,32);
    printf("==== %s ====\n", fail?"FAIL":"PASS");
    return fail?1:0;
}
