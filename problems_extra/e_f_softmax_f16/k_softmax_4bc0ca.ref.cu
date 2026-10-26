#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(128) void k_softmax_4bc0ca(const __half* x, __half* softmax) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 128;
        int a1_co = (_gid) % 128;
        float acc1 = 0.0f;
        __half acc0 = __float2half(-1e+30f);
        for (int a1 = a1_co; a1 < 8; a1 += 128) {
            float one = 1.0f;
            __half in0 = x[a0 * 8 + a1];
            float in0__gn = fmaxf(__half2float(acc0), __half2float(in0));
            float in0__dg = __half2float(acc0) - in0__gn;
            float in0__dg_o = __half2float(in0) - in0__gn;
            float in0__acc1_sa = acc1 * expf(in0__dg);
            float in0__acc1_sb = expf(in0__dg_o) * one;
            acc1 = in0__acc1_sa + in0__acc1_sb;
            acc0 = __hmax(acc0, in0);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 16);
            float acc0__o__gn = fmaxf(((float)(acc0)), acc0__o);
            float acc0__o__dg = ((float)(acc0)) - acc0__o__gn;
            float acc0__o__alpha = expf(acc0__o__dg);
            float acc0__o__dg_o = acc0__o - acc0__o__gn;
            float acc0__o__beta = expf(acc0__o__dg_o);
            float acc0__o__acc1_sa = acc0__o__alpha * acc1;
            float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
            acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
            acc0 = acc0__o__gn;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 8);
            float acc0__o__gn = fmaxf(((float)(acc0)), acc0__o);
            float acc0__o__dg = ((float)(acc0)) - acc0__o__gn;
            float acc0__o__alpha = expf(acc0__o__dg);
            float acc0__o__dg_o = acc0__o - acc0__o__gn;
            float acc0__o__beta = expf(acc0__o__dg_o);
            float acc0__o__acc1_sa = acc0__o__alpha * acc1;
            float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
            acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
            acc0 = acc0__o__gn;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 4);
            float acc0__o__gn = fmaxf(((float)(acc0)), acc0__o);
            float acc0__o__dg = ((float)(acc0)) - acc0__o__gn;
            float acc0__o__alpha = expf(acc0__o__dg);
            float acc0__o__dg_o = acc0__o - acc0__o__gn;
            float acc0__o__beta = expf(acc0__o__dg_o);
            float acc0__o__acc1_sa = acc0__o__alpha * acc1;
            float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
            acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
            acc0 = acc0__o__gn;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 2);
            float acc0__o__gn = fmaxf(((float)(acc0)), acc0__o);
            float acc0__o__dg = ((float)(acc0)) - acc0__o__gn;
            float acc0__o__alpha = expf(acc0__o__dg);
            float acc0__o__dg_o = acc0__o - acc0__o__gn;
            float acc0__o__beta = expf(acc0__o__dg_o);
            float acc0__o__acc1_sa = acc0__o__alpha * acc1;
            float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
            acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
            acc0 = acc0__o__gn;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 1);
            float acc0__o__gn = fmaxf(((float)(acc0)), acc0__o);
            float acc0__o__dg = ((float)(acc0)) - acc0__o__gn;
            float acc0__o__alpha = expf(acc0__o__dg);
            float acc0__o__dg_o = acc0__o - acc0__o__gn;
            float acc0__o__beta = expf(acc0__o__dg_o);
            float acc0__o__acc1_sa = acc0__o__alpha * acc1;
            float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
            acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
            acc0 = acc0__o__gn;
        }
        __shared__ float acc0_smem[4];
        __shared__ float acc1_smem[4];
        if (lane == 0) {
            acc0_smem[warp] = __half2float(acc0);
            acc1_smem[warp] = acc1;
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc0 = acc0_smem[warp];
                float acc0__o = acc0_smem[warp + s];
                float acc1 = acc1_smem[warp];
                float acc1__o = acc1_smem[warp + s];
                float acc0__o__gn = fmaxf(acc0, acc0__o);
                float acc0__o__dg = acc0 - acc0__o__gn;
                float acc0__o__alpha = expf(acc0__o__dg);
                float acc0__o__dg_o = acc0__o - acc0__o__gn;
                float acc0__o__beta = expf(acc0__o__dg_o);
                float acc0__o__acc1_sa = acc0__o__alpha * acc1;
                float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
                acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
                acc0 = acc0__o__gn;
                acc0_smem[warp] = acc0;
                acc1_smem[warp] = acc1;
            }
            __syncthreads();
        }
        acc0 = acc0_smem[0];
        acc1 = acc1_smem[0];
        float v2 = 1.0f / acc1;
        for (int a3 = a1_co; a3 < 8; a3 += 128) {
            __half in2 = x[a0 * 8 + a3];
            float v5 = v2 * __half2float(expf((in2 - acc0)));
            softmax[a0 * 8 + a3] = __float2half(v5);
        }
}
