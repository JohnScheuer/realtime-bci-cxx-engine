#ifndef CUDA_BCI_DECODER_CUH
#define CUDA_BCI_DECODER_CUH

#include <cuda_runtime.h>
#include <cstddef>

namespace bci {

class CudaBciDecoder {
public:
    static constexpr size_t CHANNELS = 128;
    static constexpr size_t OUTPUT_CLASSES = 8;

    CudaBciDecoder();
    ~CudaBciDecoder();

    CudaBciDecoder(const CudaBciDecoder&) = delete;
    CudaBciDecoder& operator=(const CudaBciDecoder&) = delete;

    void decode_frame(const float* host_signal_input, float* host_output);

private:
    float* host_mapped_signal_{nullptr};
    float* device_signal_ptr_{nullptr};

    float* host_mapped_output_{nullptr};
    float* device_output_ptr_{nullptr};

    float* device_weights_ptr_{nullptr};
    float* device_bias_ptr_{nullptr};

    cudaStream_t stream_;
};

} // namespace bci

#endif // CUDA_BCI_DECODER_CUH
