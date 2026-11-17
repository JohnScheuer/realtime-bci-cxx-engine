#ifndef LOCKFREE_RING_BUFFER_HPP
#define LOCKFREE_RING_BUFFER_HPP

#include <atomic>
#include <cstddef>
#include <array>
#include <new>

namespace bci {

// Single-Producer Single-Consumer (SPSC) Lock-Free Ring Buffer for Sub-Millisecond BCI Streams
template <typename T, size_t Capacity>
class LockFreeRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2 for fast bitwise masking.");

public:
    LockFreeRingBuffer() : head_(0), tail_(0) {}

    template <typename... Args>
    bool push(Args&&... args) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t current_head = head_.load(std::memory_order_acquire);

        if ((current_tail - current_head) >= Capacity) {
            return false; // Buffer full (Overrun)
        }

        buffer_[current_tail & MASK] = T(std::forward<Args>(args)...);
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t current_tail = tail_.load(std::memory_order_acquire);

        if (current_head == current_tail) {
            return false; // Buffer empty (Underrun)
        }

        item = std::move(buffer_[current_head & MASK]);
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    size_t size() const {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t tail = tail_.load(std::memory_order_relaxed);
        return (tail >= head) ? (tail - head) : 0;
    }

    bool empty() const {
        return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed);
    }

private:
    static constexpr size_t MASK = Capacity - 1;

    // Cache-line aligned buffer (64 bytes) to eliminate false sharing
    alignas(64) std::array<T, Capacity> buffer_;

    // Separate head and tail onto distinct cache lines
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
};

} // namespace bci

#endif // LOCKFREE_RING_BUFFER_HPP
