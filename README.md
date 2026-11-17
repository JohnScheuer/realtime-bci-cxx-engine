# Real-Time Sub-Millisecond C++20 / CUDA Zero-Copy BCI Inference Engine

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=for-the-badge&logo=c%2B%2B" alt="C++20">
  <img src="https://img.shields.io/badge/CUDA-12.8-green.svg?style=for-the-badge&logo=nvidia" alt="CUDA 12.8">
  <img src="https://img.shields.io/badge/Hardware-AVX2%20SIMD-orange.svg?style=for-the-badge" alt="AVX2 SIMD">
  <img src="https://img.shields.io/badge/SLA-Sub--Millisecond%20P99-brightgreen.svg?style=for-the-badge" alt="Sub-Millisecond SLA">
  <img src="https://img.shields.io/badge/Architecture-Zero--Copy%20Mapped-red.svg?style=for-the-badge" alt="Zero-Copy Architecture">
  <img src="https://img.shields.io/badge/License-MIT-purple.svg?style=for-the-badge" alt="MIT License">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20WSL2-lightgrey.svg?style=for-the-badge&logo=ubuntu" alt="Platform Linux/WSL2">
</p>

**Author:** João Felipe de Souza ([github.com/JohnScheuer](https://github.com/JohnScheuer)) — *Staff ML Systems Engineer*

---

## 1. Executive Summary & Hardware Engineering Overview

Closed-loop Brain-Computer Interfaces (BCIs)—such as intracortical microelectrode arrays recording from primary motor cortex ($M1$) or somatosensory cortex ($S1$)—require **deterministic sub-millisecond end-to-end inference latency**. Biological neural processing operates on temporal windows where tactile sensory feedback and motor command loops break down if processing jitter exceeds 1.0 ms (1000 µs).

Standard Python/PyTorch inference pipelines suffer from Global Interpreter Lock (GIL) contention, non-deterministic dynamic memory allocations, garbage collection pauses, and multi-millisecond PCIe buffer transfer latency (`cudaMemcpyHostToDevice` / `cudaMemcpyDeviceToHost`).

This repository provides a production-grade **C++20 / CUDA Zero-Copy Low-Latency BCI Inference Engine** engineered specifically for 128-channel motor cortical arrays running at 1000 Hz closed-loop throughput under strict microsecond Service Level Agreements (SLAs).

### Key Architectural Highlights:
1. **Zero-Copy Host-Mapped CUDA Memory (`cudaHostAllocMapped`):** Eliminates PCIe host-to-device memory copy overhead by pinning host memory directly into the GPU unified address space via `cudaHostGetDevicePointer`.
2. **Lock-Free SPSC Ring Buffer (`alignas(64)`):** Single-Producer Single-Consumer lock-free ring buffer utilizing C++20 atomic acquire-release semantics and explicit cache-line alignment to eliminate false sharing between signal acquisition and GPU decoding threads.
3. **AVX2 SIMD Common Average Reference (CAR) Filtering:** Exploits 256-bit AVX2 vector registers (`__m256`) to perform spatial mean subtraction across 128 microelectrode channels in sub-microsecond CPU time (**~0.04 µs / 40 ns per frame** in L1 cache).
4. **Active-Spin Stream Scheduling (`cudaDeviceScheduleSpin`):** Forces the CUDA runtime into active-spin polling, eliminating operating system kernel context switches and scheduler quantum yield stalls.
5. **Deterministic CUDA GEMV Kernel:** Custom linear projection GEMV kernel executing motor kinematic intent extraction asynchronously within dedicated CUDA streams (**~60 µs per frame**).
6. **Microsecond Precision SLA Profiling:** Real-time percentile profiling tracking Min, P50, P90, P99, P99.9, and Max latencies against a strict sub-millisecond ($< 1000\text{ µs}$) SLA.

---

## 2. Empirical Latency Benchmarks & Diagnostic Figures

The benchmark was executed on an **NVIDIA GeForce RTX 5060 (8GB GDDR7, SM 12.0)** and an **AMD Ryzen 5 5600X CPU** running **Ubuntu 22.04 LTS (WSL2)** under C++20 (`g++ 11.4.0`) and CUDA 12.8. 

A total of **10,000 continuous neural frames** (10 seconds at 1000 Hz) were processed in real time.

### Empirical Benchmark Results Table

| Latency Metric | Measured Value (µs) | SLA Target Limit (µs) | Headroom / SLA Compliance |
| :--- | :---: | :---: | :---: |
| **Minimum Latency** | **39.45 µs** | 1000.00 µs | **25.3x Below Target** |
| **Mean Latency** | **62.67 µs** | 1000.00 µs | **15.9x Below Target** |
| **P50 Latency (Median)** | **52.15 µs** | 1000.00 µs | **19.1x Below Target** |
| **P90 Latency** | **79.71 µs** | 1000.00 µs | **12.5x Below Target** |
| **P99 Latency (SLA Metric)** | **183.91 µs** | **1000.00 µs** | **5.4x Headroom (PASS)** |
| **P99.9 Latency** | **299.58 µs** | 1000.00 µs | **3.3x Headroom (PASS)** |
| **Maximum Latency** | **330.63 µs** | 1000.00 µs | **3.0x Headroom (PASS)** |
| **SLA Violation Rate** | **0.0000 %** | $< 0.0100 \%$ | **100% Deterministic Guarantee** |

---

### A. Real-Time Latency Trace Across 10,000 Stream Frames
![Continuous Latency Trace](figures/latency_timeline.png)

### B. Latency Probability Distribution & Cumulative Distribution Function (CDF)
![Latency Histogram and CDF](figures/latency_distribution.png)

### C. Pipeline Stage Latency Breakdown (AVX2 CAR vs CUDA Zero-Copy GEMV)
![Stage Execution Time Breakdown](figures/stage_breakdown.png)

### D. 128-Channel Neural Signal Common Average Reference (CAR) Filtering
![AVX2 CAR Neural Waveform Processing](figures/neural_signal_car.png)

---

## 3. Architecture & Pipeline Topology

```
+-----------------------------------------------------------------------------------+
|                        1000 Hz Cortical Signal Stream                             |
|              128 Microelectrode Channels (Action Potential Signals)                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       Producer Thread (Acquisition Simulator)                     |
|                   Pacing: 1000 µs intervals (Microsecond Precision)               |
+-----------------------------------------------------------------------------------+
                                         |
            Lock-Free SPSC Ring Buffer (C++20 std::atomic, alignas(64))
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|               Consumer Thread: Sub-Millisecond BCI Inference Engine               |
|                                                                                   |
|  [Step 1] AVX2 SIMD Spatial CAR Filter                                            |
|          Spatial Mean Calculation & Subtraction across 128 Channels               |
|          Executes on CPU in ~0.04 µs (40 ns) via 256-bit AVX2 intrinsics in L1     |
|                                                                                   |
|  [Step 2] Zero-Copy Mapped Memory Transfer                                        |
|          Direct host memory write to pinned mapped region                         |
|          Bypasses explicit PCIe cudaMemcpy transfers                              |
|                                                                                   |
|  [Step 3] CUDA Parallel GEMV Motor Decoder Kernel                                 |
|          Active-Spin Scheduling (cudaDeviceScheduleSpin)                          |
|          Decodes 128 channels -> 8 spatial trajectory / velocity classes in ~60µs |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                      Microsecond Latency Profiler & CSV Logger                    |
|             Calculates Empirical P50, P90, P99, P99.9 Latency Percentiles           |
+-----------------------------------------------------------------------------------+
```

---

## 4. C++20 & CUDA Implementation Highlights

### A. Lock-Free SPSC Ring Buffer (`include/lockfree_ring_buffer.hpp`)
```cpp
// Cache-line alignment (64 bytes) prevents false sharing between producer and consumer threads
template <typename T, size_t Capacity>
class LockFreeRingBuffer {
private:
    alignas(64) std::array<T, Capacity> buffer_;
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
};
```

### B. AVX2 Common Average Reference (CAR) (`include/avx_signal_processor.hpp`)
```cpp
// Compute spatial average across 128 channels using AVX2 SIMD 256-bit registers (~40 ns)
__m256 sum_vec = _mm256_setzero_ps();
for (size_t i = 0; i < CHANNELS; i += 8) {
    __m256 vec = _mm256_loadu_ps(input + i);
    sum_vec = _mm256_add_ps(sum_vec, vec);
}
```

### C. Zero-Copy Mapped CUDA Host Memory (`src/cuda_bci_decoder.cu`)
```cpp
// Allocate host-mapped memory accessible directly by GPU without PCIe memcpy overhead
cudaHostAlloc((void**)&host_mapped_signal_, CHANNELS * sizeof(float), cudaHostAllocMapped);
cudaHostGetDevicePointer((void**)&device_signal_ptr_, host_mapped_signal_, 0);
```

---

## 5. Build & Execution Guide

### Prerequisites
- Linux / WSL2 Ubuntu 22.04 LTS
- `g++` 11.4+ (supporting C++20)
- NVIDIA CUDA Toolkit 12.0+ & NVCC
- CMake 3.20+
- Python 3.10+ with `pandas`, `numpy`, and `matplotlib`

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/JohnScheuer/realtime-bci-cxx-engine.git
cd realtime-bci-cxx-engine

# 2. Configure & Build C++20 Engine
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. Execute Real-Time Sub-Millisecond Benchmark
./build/realtime_bci_engine

# 4. Generate Diagnostic Benchmark Figures
python3 scripts/generate_figures.py
```

---

## 6. Deterministic Ultra-Low Latency Implications (HFT, Robotics & Real-Time Streaming)

Beyond neural decoding pipelines, the zero-copy pinned host memory architecture, cache-aligned SPSC lock-free ring buffers, and active-spin CUDA stream scheduling engineered in this runtime solve foundational infrastructure bottlenecks in mission-critical real-time systems:

1. **Zero-Jitter LLM Token Streaming Systems:** Eliminating PCIe `cudaMemcpy` overhead and OS kernel context-switch stalls provides the exact microsecond determinism required to stream tokens from large language models without inter-token jitter.
2. **High-Frequency Trading (HFT) Market Data Processing:** Sub-microsecond AVX2 SIMD signal processing coupled with active-spin stream polling delivers sub-100 microsecond processing guarantees essential for high-throughput order routing and risk checks.
3. **High-Rate Autonomous Robotics & Closed-Loop Control:** Operating at **52.15 µs median latency** leaves $> 940\text{ µs}$ of headroom within a 1.0 ms control loop for complex kinematics solver execution, sensor fusion, and real-time actuator feedback control.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
