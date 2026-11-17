#include "cuda_bci_decoder.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <random>

namespace bci {

__global__ void gemv_decoder_kernel(const float* __restrict__ signal,
                                    const float* __restrict__ weights,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int channels, int outputs) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx < outputs) {
        float sum = bias[out_idx];
        for (int c = 0; c < channels; ++c) {
            sum += signal[c] * weights[out_idx * channels + c];
        }
        output[out_idx] = sum;
    }
}

CudaBciDecoder::CudaBciDecoder() {
    cudaError_t err = cudaHostAlloc((void**)&host_mapped_signal_, CHANNELS * sizeof(float), cudaHostAllocMapped);
    if (err != cudaSuccess) throw std::runtime_error("cudaHostAlloc signal failed");
    
    err = cudaHostGetDevicePointer((void**)&device_signal_ptr_, host_mapped_signal_, 0);
    if (err != cudaSuccess) throw std::runtime_error("cudaHostGetDevicePointer signal failed");

    err = cudaHostAlloc((void**)&host_mapped_output_, OUTPUT_CLASSES * sizeof(float), cudaHostAllocMapped);
    if (err != cudaSuccess) throw std::runtime_error("cudaHostAlloc output failed");

    err = cudaHostGetDevicePointer((void**)&device_output_ptr_, host_mapped_output_, 0);
    if (err != cudaSuccess) throw std::runtime_error("cudaHostGetDevicePointer output failed");

    cudaMalloc((void**)&device_weights_ptr_, OUTPUT_CLASSES * CHANNELS * sizeof(float));
    cudaMalloc((void**)&device_bias_ptr_, OUTPUT_CLASSES * sizeof(float));

    std::vector<float> h_weights(OUTPUT_CLASSES * CHANNELS);
    std::vector<float> h_bias(OUTPUT_CLASSES, 0.0f);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (auto& w : h_weights) w = dist(rng);

    cudaMemcpy(device_weights_ptr_, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_bias_ptr_, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaStreamCreate(&stream_);
}

CudaBciDecoder::~CudaBciDecoder() {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);

    if (device_weights_ptr_) cudaFree(device_weights_ptr_);
    if (device_bias_ptr_) cudaFree(device_bias_ptr_);
    if (host_mapped_signal_) cudaFreeHost(host_mapped_signal_);
    if (host_mapped_output_) cudaFreeHost(host_mapped_output_);
}

void CudaBciDecoder::decode_frame(const float* host_signal_input, float* host_output) {
    // Direct zero-copy host memory write
    std::memcpy(host_mapped_signal_, host_signal_input, CHANNELS * sizeof(float));

    int threads_per_block = 32;
    int blocks = (OUTPUT_CLASSES + threads_per_block - 1) / threads_per_block;

    gemv_decoder_kernel<<<blocks, threads_per_block, 0, stream_>>>(
        device_signal_ptr_,
        device_weights_ptr_,
        device_bias_ptr_,
        device_output_ptr_,
        CHANNELS,
        OUTPUT_CLASSES
    );

    // Active spin stream synchronization
    cudaStreamSynchronize(stream_);

    // Direct zero-copy read
    std::memcpy(host_output, host_mapped_output_, OUTPUT_CLASSES * sizeof(float));
}

} // namespace bci
