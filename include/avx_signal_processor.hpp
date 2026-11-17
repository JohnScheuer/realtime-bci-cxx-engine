#ifndef AVX_SIGNAL_PROCESSOR_HPP
#define AVX_SIGNAL_PROCESSOR_HPP

#include <immintrin.h>
#include <cstddef>
#include <numeric>
#include <vector>

namespace bci {

// AVX2 SIMD Accelerated Common Average Reference (CAR) Signal Filter
class AVXSignalProcessor {
public:
    static constexpr size_t CHANNELS = 128;

    // Process frame in-place or write to output array: input[128] -> output[128]
    static void process_car(const float* __restrict__ input, float* __restrict__ output) {
        // Step 1: Compute total spatial average across 128 channels using AVX2 SIMD
        __m256 sum_vec = _mm256_setzero_ps();

        // 128 floats = 16 AVX2 vectors (8 floats per __m256)
        for (size_t i = 0; i < CHANNELS; i += 8) {
            __m256 vec = _mm256_loadu_ps(input + i);
            sum_vec = _mm256_add_ps(sum_vec, vec);
        }

        // Horizontal sum across __m256 register
        alignas(32) float temp[8];
        _mm256_storeu_ps(temp, sum_vec);
        float total_sum = 0.0f;
        for (int i = 0; i < 8; ++i) {
            total_sum += temp[i];
        }

        const float avg = total_sum / static_cast<float>(CHANNELS);
        __m256 avg_vec = _mm256_set1_ps(avg);

        // Step 2: Subtract spatial mean from each channel signal (Common Average Reference)
        for (size_t i = 0; i < CHANNELS; i += 8) {
            __m256 vec = _mm256_loadu_ps(input + i);
            __m256 car_vec = _mm256_sub_ps(vec, avg_vec);
            _mm256_storeu_ps(output + i, car_vec);
        }
    }
};

} // namespace bci

#endif // AVX_SIGNAL_PROCESSOR_HPP
