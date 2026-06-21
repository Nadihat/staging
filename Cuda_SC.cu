/*
    ServitorConnect Web - CUDA Translation
    Executes the exact 88,888,888 polynomial hash chain natively on the GPU.
    
    To compile: 
    nvcc -O3 ServitorConnect_CUDA.cu -o ServitorConnect_CUDA.exe
    
    Usage:
    ServitorConnect_CUDA.exe -i "I am love" -r 88888 -d 3600
*/

#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

// Macro for capturing CUDA errors
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
      if (abort) exit(code);
   }
}

// ---------------------------------------------------------
// THE CUDA KERNEL
// Runs entirely inside the GPU's ultra-fast internal registers
// ---------------------------------------------------------
__global__ void ServitorConnectKernel(unsigned int initial_hash, unsigned long long repeats, unsigned long long* global_accumulator) 
{
    // Calculate which "mind" in the choir this thread represents
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= repeats) return;

    unsigned int h = initial_hash;
    char buf[10]; // Tiny 10-byte buffer physically on the GPU core

    // The inner loop: Exactly 88,888,888 total hashes. 
    // We start at 1 because the CPU already did i=0 (the base text hash)
    for (int i = 1; i < 88888888; i++) 
    {
        // 1. FAST INTEGER-TO-STRING (Pure Math, No Objects)
        unsigned int temp = h;
        int len = 0;
        
        if (temp == 0) {
            buf[len++] = '0';
        } else {
            // Extracts digits in reverse order
            while (temp > 0) {
                buf[len++] = (temp % 10) + '0';
                temp /= 10;
            }
        }

        // 2. POLYNOMIAL ROLLING HASH (Matches Javascript's >>> 0 exact math)
        h = 0;
        // Read the digits backwards to match the correct forward string order
        for (int j = len - 1; j >= 0; j--) {
            h = h * 31 + buf[j];
        }
    }

    // THE ANTI-OPTIMIZATION TRICK:
    // We write the final state to a global accumulator exactly ONCE after 88,888,888 iterations.
    // This forces the compiler to actually do the work, while causing zero performance bottlenecks.
    atomicAdd(global_accumulator, (unsigned long long)h);
}

// ---------------------------------------------------------
// MAIN APPLICATION
// ---------------------------------------------------------
int main(int argc, char **argv)
{
    std::string intention = "I am calm and balanced.";
    unsigned long long repeats = 88888; // Default to 88,888 simultaneous GPU threads
    int duration = 3600; // 1 Hour

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) intention = argv[++i];
        else if (arg == "-r" && i + 1 < argc) repeats = std::stoull(argv[++i]);
        else if (arg == "-d" && i + 1 < argc) duration = std::stoi(argv[++i]);
    }

    std::cout << "==========================================\n";
    std::cout << "      ServitorConnect CUDA Translation    \n";
    std::cout << "==========================================\n";
    std::cout << "Intention: " << intention << "\n";
    std::cout << "Repeats (Simultaneous Threads): " << repeats << " per hour\n";
    std::cout << "Duration: " << duration << " seconds\n\n";

    // 1. Calculate the initial string hash on the CPU exactly once.
    // This perfectly mimics JS charCodeAt(j) for standard text.
    unsigned int initial_hash = 0;
    for (size_t i = 0; i < intention.length(); i++) {
        initial_hash = initial_hash * 31 + (unsigned char)intention[i];
    }

    // 2. Allocate the anti-optimization global accumulator on the GPU
    unsigned long long* d_accumulator;
    cudaCheckError(cudaMalloc(&d_accumulator, sizeof(unsigned long long)));

    // 3. Setup CUDA Grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (repeats + threadsPerBlock - 1) / threadsPerBlock;

    // Time tracking
    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::seconds(duration);
    auto next_batch_time = start_time;

    // Loop until duration expires
    while (true) 
    {
        auto current_time = std::chrono::steady_clock::now();
        
        if (current_time >= end_time) {
            break;
        }

        // Trigger the hourly batch on the GPU
        if (current_time >= next_batch_time) {
            std::cout << "\n[->] Launching GPU Choir (" << repeats << " threads doing 88,888,888 iterations)...\n";
            
            // Reset accumulator
            cudaCheckError(cudaMemset(d_accumulator, 0, sizeof(unsigned long long)));

            // Fire the GPU kernels
            ServitorConnectKernel<<<blocksPerGrid, threadsPerBlock>>>(initial_hash, repeats, d_accumulator);
            
            // Wait for GPU to finish
            cudaCheckError(cudaDeviceSynchronize());
            
            std::cout << "[✓] GPU Batch Complete. Reality impacted.\n\n";

            next_batch_time += std::chrono::seconds(3600); // Add 1 hour
        }

        // Print visual countdown (refreshes every 1 second)
        auto remaining_sec = std::chrono::duration_cast<std::chrono::seconds>(end_time - current_time).count();
        int hrs = remaining_sec / 3600;
        int mins = (remaining_sec % 3600) / 60;
        int secs = remaining_sec % 60;

        printf("\rStatus: Intent Repeated %llu Times Hourly | Remaining: %02d:%02d:%02d", repeats, hrs, mins, secs);
        fflush(stdout);

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "\n\nProcess completed natively on the GPU.\n";

    // Clean up
    cudaFree(d_accumulator);
    return 0;
}
