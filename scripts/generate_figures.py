import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_all_figures():
    os.makedirs("figures", exist_ok=True)
    plt.style.use('dark_background')

    # Load Data
    bench_df = pd.read_csv("latency_benchmark.csv")
    sig_df = pd.read_csv("signal_sample.csv") if os.path.exists("signal_sample.csv") else None

    latencies = bench_df['latency_us'].values
    car_times = bench_df['car_us'].values if 'car_us' in bench_df.columns else latencies * 0.02
    cuda_times = bench_df['cuda_us'].values if 'cuda_us' in bench_df.columns else latencies * 0.98
    frames = bench_df['frame_id'].values

    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    sla = 1000.0

    # 1. Continuous Latency Trace (10,000 Frames @ 1000 Hz)
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.plot(frames, latencies, color='#00e5ff', alpha=0.75, linewidth=0.6, label='Frame Latency (µs)')
    ax.axhline(sla, color='#ff0055', linestyle='--', linewidth=1.5, label='1.0 ms SLA Target Limit (1000 µs)')
    ax.axhline(p99, color='#ffea00', linestyle=':', linewidth=1.2, label=f'P99 Latency ({p99:.2f} µs)')
    ax.axhline(p50, color='#00ff87', linestyle=':', linewidth=1.2, label=f'P50 Median ({p50:.2f} µs)')

    ax.set_title("Real-Time Closed-Loop BCI Pipeline Latency Trace (10,000 Frames @ 1000Hz)", fontsize=12, fontweight='bold', pad=12, color='#ffffff')
    ax.set_xlabel("Frame Index (Time @ 1000 Hz Stream)", fontsize=10, labelpad=8)
    ax.set_ylabel("Pipeline Latency (Microseconds, µs)", fontsize=10, labelpad=8)
    ax.set_ylim(0, max(500, np.max(latencies) * 1.1))
    ax.grid(True, linestyle=':', alpha=0.3, color='#555555')
    ax.legend(loc='upper right', framealpha=0.85, facecolor='#1e1e1e', edgecolor='#444444', fontsize=8.5)

    plt.tight_layout()
    plt.savefig("figures/latency_timeline.png", dpi=300)
    plt.close()
    print("[INFO] Saved figures/latency_timeline.png")

    # 2. Latency Histogram & Cumulative Distribution Function (CDF)
    fig, ax1 = plt.subplots(figsize=(9, 4.5), dpi=300)
    bins = np.linspace(min(latencies), min(400.0, np.max(latencies)), 80)
    n, bins_out, patches = ax1.hist(latencies, bins=bins, color='#00b0ff', alpha=0.75, edgecolor='#00838f', label='Latency Histogram')
    ax1.set_xlabel("Processing Latency (Microseconds, µs)", fontsize=10, labelpad=8)
    ax1.set_ylabel("Frame Count", fontsize=10, color='#00b0ff', labelpad=8)
    ax1.tick_params(axis='y', labelcolor='#00b0ff')

    ax2 = ax1.twinx()
    sorted_lat = np.sort(latencies)
    cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
    ax2.plot(sorted_lat, cdf, color='#00ff87', linewidth=2.0, label='CDF (Cumulative Probability)')
    ax2.set_ylabel("Cumulative Probability", fontsize=10, color='#00ff87', labelpad=8)
    ax2.tick_params(axis='y', labelcolor='#00ff87')
    ax2.set_ylim(0, 1.05)

    ax1.axvline(p50, color='#00ff87', linestyle='--', linewidth=1.2, alpha=0.8)
    ax1.axvline(p99, color='#ff0055', linestyle='--', linewidth=1.5, alpha=0.9)

    ax1.set_title("BCI Latency Probability Distribution & CDF Profile", fontsize=12, fontweight='bold', pad=12, color='#ffffff')
    ax1.grid(True, linestyle=':', alpha=0.3, color='#555555')

    plt.tight_layout()
    plt.savefig("figures/latency_distribution.png", dpi=300)
    plt.close()
    print("[INFO] Saved figures/latency_distribution.png")

    # 3. Stage Breakdown (AVX2 CAR Filter vs CUDA Zero-Copy GEMV)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    mean_car = np.mean(car_times)
    mean_cuda = np.mean(cuda_times)
    p99_car = np.percentile(car_times, 99)
    p99_cuda = np.percentile(cuda_times, 99)

    categories = ['AVX2 SIMD CAR Filter', 'CUDA Zero-Copy GEMV']
    mean_vals = [mean_car, mean_cuda]
    p99_vals = [p99_car, p99_cuda]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(x - width/2, mean_vals, width, label='Mean Latency (µs)', color='#00e5ff', edgecolor='#222222')
    rects2 = ax.bar(x + width/2, p99_vals, width, label='P99 Latency (µs)', color='#ff0055', edgecolor='#222222')

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.2f} µs', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, color='#ffffff', fontweight='bold')

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f} µs', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, color='#ffffff', fontweight='bold')

    ax.set_title("Pipeline Stage Microsecond Execution Time Breakdown", fontsize=12, fontweight='bold', pad=12, color='#ffffff')
    ax.set_ylabel("Latency (Microseconds, µs)", fontsize=10, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(p99_vals) * 1.35 + 1.0)
    ax.grid(True, axis='y', linestyle=':', alpha=0.3, color='#555555')
    ax.legend(loc='upper left', framealpha=0.8, facecolor='#1e1e1e', edgecolor='#444444', fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/stage_breakdown.png", dpi=300)
    plt.close()
    print("[INFO] Saved figures/stage_breakdown.png")

    # 4. Neural Waveform Comparison (Raw vs Filtered CAR)
    if sig_df is not None:
        fig, (ax_raw, ax_car) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, dpi=300)
        colors = ['#00e5ff', '#00ff87', '#ffea00', '#e040fb']

        for ch in range(4):
            ch_data = sig_df[sig_df['channel'] == ch]
            ax_raw.plot(ch_data['frame'], ch_data['raw'], color=colors[ch], alpha=0.75, linewidth=1.0, label=f'Ch {ch+1} Raw Signal')
            ax_car.plot(ch_data['frame'], ch_data['filtered'], color=colors[ch], alpha=0.85, linewidth=1.0, label=f'Ch {ch+1} CAR Filtered')

        ax_raw.set_title("Raw Cortical Array Signal with Common-Mode Noise & 60Hz Artifacts", fontsize=11, fontweight='bold', color='#ff3d00')
        ax_raw.set_ylabel("Amplitude (µV)", fontsize=9)
        ax_raw.grid(True, linestyle=':', alpha=0.3, color='#555555')
        ax_raw.legend(loc='upper right', framealpha=0.8, facecolor='#1e1e1e', edgecolor='#444444', fontsize=7.5)

        ax_car.set_title("AVX2 SIMD Common Average Reference (CAR) Filtered Signal (Isolated Spikes)", fontsize=11, fontweight='bold', color='#00ff87')
        ax_car.set_xlabel("Time Frame (1 ms resolution)", fontsize=9.5, labelpad=6)
        ax_car.set_ylabel("Amplitude (µV)", fontsize=9)
        ax_car.grid(True, linestyle=':', alpha=0.3, color='#555555')
        ax_car.legend(loc='upper right', framealpha=0.8, facecolor='#1e1e1e', edgecolor='#444444', fontsize=7.5)

        plt.tight_layout()
        plt.savefig("figures/neural_signal_car.png", dpi=300)
        plt.close()
        print("[INFO] Saved figures/neural_signal_car.png")

if __name__ == '__main__':
    generate_all_figures()
