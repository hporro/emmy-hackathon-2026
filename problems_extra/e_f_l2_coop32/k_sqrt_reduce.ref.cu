extern "C" __global__
__launch_bounds__(32) void k_sqrt_reduce(const float* x, float* sqrt) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 32;
        int a1_co = (_gid) % 32;
        float acc0 = 0.0f;
        for (int a1 = a1_co; a1 < 1024; a1 += 32) {
            float in0 = x[a0 * 1024 + a1];
            float v0 = in0 * in0;
            acc0 += v0;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            acc0 = acc0 + acc0__o;
        }
        if (a1_co == 0) {
            float v1 = sqrtf(acc0);
            sqrt[a0] = v1;
        }
}
