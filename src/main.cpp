#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <random>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "lockfree_ring_buffer.hpp"
#include "avx_signal_processor.hpp"
#include "cuda_bci_decoder.cuh"
#include "latency_profiler.hpp"

using namespace bci;

struct NeuralFrame {
    uint64_t timestamp_ns;
    float channels[128];
};

struct FrameTiming {
    size_t frame_id;
    double car_us;
    double total_us;
};

struct SignalLogSample {
    size_t frame;
    size_t channel;
    float raw;
    float filtered;
};

static std::atomic<bool> g_running{true};

// Producer Thread: Continuous high-frequency neural signal streamer
void producer_thread_fn(LockFreeRingBuffer<NeuralFrame, 1024>& ring_buffer, size_t total_frames) {
    std::mt19937 rng(1337);
    std::normal_distribution<float> noise(0.0f, 12.0f);
    std::normal_distribution<float> common_noise(0.0f, 40.0f);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < total_frames && g_running.load(std::memory_order_relaxed); ++i) {
        NeuralFrame frame;
        auto now = std::chrono::high_resolution_clock::now();
        frame.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time).count();

        float cm_artifact = common_noise(rng) + 35.0f * std::sin(2.0 * 3.1415926535 * 60.0 * (i / 1000.0));

        for (size_t ch = 0; ch < 128; ++ch) {
            float ap_spike = 0.0f;
            if ((i + ch * 7) % 60 == 0) {
                ap_spike = -110.0f;
            } else if ((i + ch * 7) % 60 == 1) {
                ap_spike = 55.0f;
            }
            frame.channels[ch] = noise(rng) + cm_artifact + ap_spike;
        }

        while (!ring_buffer.push(frame) && g_running.load(std::memory_order_relaxed)) {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
        }
    }
}

int main() {
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);

    std::cout << "======================================================================\n";
    std::cout << " Real-Time Sub-Millisecond C++20/CUDA Zero-Copy BCI Inference Engine \n";
    std::cout << " Target SLA: < 1.000 ms per 128-channel packet (1000 Hz closed-loop) \n";
    std::cout << " Mode: High-Throughput Continuous Stream (Zero OS Timer Slack)      \n";
    std::cout << "======================================================================\n";

    constexpr size_t TOTAL_FRAMES = 10000;
    LockFreeRingBuffer<NeuralFrame, 1024> ring_buffer;
    LatencyProfiler profiler(1000.0);

    std::cout << "[INFO] Initializing CUDA Zero-Copy Context & Allocating Pinned Memory...\n";
    CudaBciDecoder decoder;

    // Warmup CUDA context to ensure active P-State (P0)
    alignas(32) float warmup_input[128] = {0.0f};
    float warmup_output[8] = {0.0f};
    for (int i = 0; i < 100; ++i) {
        decoder.decode_frame(warmup_input, warmup_output);
    }
    std::cout << "[INFO] CUDA Zero-Copy Context Warmed Up Successfully.\n";

    std::thread producer(producer_thread_fn, std::ref(ring_buffer), TOTAL_FRAMES);

    alignas(32) float car_filtered_signal[128];
    float decoded_intent[8];

    std::vector<FrameTiming> timings;
    timings.reserve(TOTAL_FRAMES);

    std::vector<SignalLogSample> signal_samples;
    signal_samples.reserve(150 * 4);

    size_t processed_frames = 0;
    std::cout << "[INFO] Starting Closed-Loop Inference Pipeline Execution...\n";

    while (processed_frames < TOTAL_FRAMES) {
        NeuralFrame frame;
        if (ring_buffer.pop(frame)) {
            auto t0 = std::chrono::high_resolution_clock::now();

            // Step 1: AVX2 SIMD Common Average Reference (CAR) filtering
            AVXSignalProcessor::process_car(frame.channels, car_filtered_signal);
            auto t1 = std::chrono::high_resolution_clock::now();

            // Step 2: CUDA Zero-Copy GEMV Decoding Kernel
            decoder.decode_frame(car_filtered_signal, decoded_intent);
            auto t2 = std::chrono::high_resolution_clock::now();

            double car_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            double total_us = std::chrono::duration<double, std::micro>(t2 - t0).count();

            profiler.record_latency(total_us);
            timings.push_back({processed_frames, car_us, total_us});

            if (processed_frames < 150) {
                for (size_t ch = 0; ch < 4; ++ch) {
                    signal_samples.push_back({processed_frames, ch, frame.channels[ch], car_filtered_signal[ch]});
                }
            }

            processed_frames++;
        } else {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
        }
    }

    g_running.store(false);
    if (producer.joinable()) {
        producer.join();
    }

    profiler.print_report("REALTIME BCI ENGINE LATENCY BENCHMARK (RTX 5060 + Ryzen 5600X)");

    std::ofstream csv_file("latency_benchmark.csv");
    csv_file << "frame_id,car_us,latency_us\n";
    for (const auto& t : timings) {
        csv_file << t.frame_id << "," << t.car_us << "," << t.total_us << "\n";
    }
    csv_file.close();

    std::ofstream raw_sig_file("signal_sample.csv");
    raw_sig_file << "frame,channel,raw,filtered\n";
    for (const auto& s : signal_samples) {
        raw_sig_file << s.frame << "," << s.channel << "," << s.raw << "," << s.filtered << "\n";
    }
    raw_sig_file.close();

    std::cout << "[INFO] Latency distribution saved to 'latency_benchmark.csv'.\n";
    std::cout << "[INFO] Signal waveform samples saved to 'signal_sample.csv'.\n";

    return 0;
}
