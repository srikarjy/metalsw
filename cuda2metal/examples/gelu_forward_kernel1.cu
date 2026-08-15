// Source: https://github.com/karpathy/llm.c/blob/master/dev/cuda/gelu_forward.cu
// (gelu_forward_kernel1 only — the naive, non-Packed128 version). floatX is llm.c's
// precision typedef; treated as float here since cuda2metal doesn't model it.
typedef float floatX;

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

__global__ void gelu_forward_kernel1(floatX* out, const floatX* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}
