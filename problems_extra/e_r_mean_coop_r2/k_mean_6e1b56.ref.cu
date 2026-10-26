extern "C" __global__
__launch_bounds__(32) void k_mean_6e1b56(const float* mean_count, const float* x, float* mean) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 32;
        int a1_co = (_gid) % 32;
        float acc0 = 0.0f;
        float acc0__r1 = 0.0f;
        for (int a1 = a1_co; a1 < 1024; a1 += 64) {
            float in1 = x[a0 * 1024 + a1];
            acc0 += in1;
            float in1__r1 = x[a0 * 1024 + (a1 + 32)];
            acc0__r1 += in1__r1;
        }
        acc0 += acc0__r1;
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
            float in0 = mean_count[0];
            float v1 = acc0 * (1.0f / in0);
            mean[a0] = v1;
        }
}
