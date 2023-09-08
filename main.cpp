#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <mutex>
#if __x86_64__
    #include <immintrin.h>
#else
    #include "sse2neon.h"
#endif

typedef unsigned long long ull;
ull iter = 1000000000;
std::mutex lock;
void one_thread_sum () {
    auto start = std::chrono::high_resolution_clock::now();
    ull sum = 0;
    for (ull i = 1; i <= iter; i++) {
        sum += i;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = (end - start);
    std::cout <<"\n1 thread time: "<<dur.count()<<" ms\nresults: " << sum << "\n";
}

void th_worker (ull frst, ull last, ull& save) {
    ull sum = 0;
    for (ull i = frst + 1; i <= last; ++i) {
        sum += i;
    }
    lock.lock();
    save += sum;
    lock.unlock();
}

void th_worker_simd (ull frst, ull last, ull& save) {
    ull sum = 0;
    __m128i __sum = _mm_setzero_si128();
    for (ull i = frst + 1; i <= last; i+=2) {
        __m128i ot = _mm_set_epi64x(i, i + 1);
        __sum = _mm_add_epi64(ot, __sum);
    }
    __m128i sll = _mm_set_epi64x(_mm_extract_epi64(__sum, 0), _mm_extract_epi64(__sum, 1));
    __m128i line_sum = _mm_add_epi64(__sum, sll);
    _mm_storeu_si64((__m128i*) &sum, line_sum);
    lock.lock();
    save += sum;
    lock.unlock();
}

void all_threads_sum () {
    int cores = std::thread::hardware_concurrency();
    ull step = iter / cores;
    ull sum = 0;
    std::vector<std::thread> ths;
    for (int i = 0; i < cores; ++i) {
        ull rem = (i == cores - 1) ? iter % cores : 0;
        ths.emplace_back(th_worker, i * step, (i + 1) * step + rem, std::ref(sum));
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (auto& th : ths) {
        th.join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = (end - start);
    std::cout << cores <<" thread time: " << dur.count() << " ms\nresults: " << sum << "\n";
}

void all_threads_sum_simd () {
    int cores = std::thread::hardware_concurrency();
    ull step = iter / cores;
    ull sum = 0;
    std::vector<std::thread> ths;
    for (int i = 0; i < cores; ++i) {
        ull rem = (i == cores - 1) ? iter % cores : 0;
        ths.emplace_back(th_worker_simd, i * step, (i + 1) * step + rem, std::ref(sum));
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (auto& th : ths) {
        th.join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = (end - start);
    std::cout << cores <<" threads + simd time: " << dur.count() << " ms\nresults: " << sum << "\n";
}

int main() {
    for (int i = 0; i < 10; ++i) {
        std::cout << "Ex. " << i << '\n';
        one_thread_sum();
        all_threads_sum();
        all_threads_sum_simd();
        std::cout << '\n';
    }
    return 0;
}
