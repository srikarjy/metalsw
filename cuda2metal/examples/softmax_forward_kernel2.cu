// Source: https://github.com/karpathy/llm.c/blob/master/dev/cuda/softmax_forward.cu
// (softmax_forward_kernel2 only — one threadgroup per row, shared-memory max reduction).
__global__ void softmax_forward_kernel2(float* out, const float* inp, int N, int C) {
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    const float* x = inp + idx * C;
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += block_size) {
        maxval = fmaxf(maxval, x[i]);
    }
    shared[tid] = maxval;
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
    }
    __syncthreads();
    float offset = shared[0];
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = expf(x[i] - offset);
    }
}
