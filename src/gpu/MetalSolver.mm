#include "gpu/MetalSolver.h"

#ifdef __APPLE__

#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <simd/simd.h>

// Definition of the Pimpl struct
struct MetalSolver::MetalImpl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLComputePipelineState> pipelineState = nil;
    id<MTLLibrary> library = nil;
};

// --- MetalSolver Implementation ---

MetalSolver::MetalSolver() : pimpl(std::make_unique<MetalImpl>()) {}

MetalSolver::~MetalSolver() {
    // The unique_ptr will automatically handle destruction.
    // We need to release the Metal objects here before the pimpl is destroyed.
    if (pimpl) {
        if (pimpl->pipelineState) [pimpl->pipelineState release];
        if (pimpl->library) [pimpl->library release];
        if (pimpl->commandQueue) [pimpl->commandQueue release];
        if (pimpl->device) [pimpl->device release];
    }
}

bool MetalSolver::init() {
    // Create Metal device
    pimpl->device = MTLCreateSystemDefaultDevice();
    if (!pimpl->device) {
        std::cerr << "MetalSolver::init() - Failed to create Metal device" << std::endl;
        return false; // No metal device found
    }
    
    // Create command queue
    pimpl->commandQueue = [pimpl->device newCommandQueue];
    if (!pimpl->commandQueue) {
        std::cerr << "MetalSolver::init() - Failed to create Metal command queue" << std::endl;
        return false;
    }

    // Create the Metal shader source
    NSString* shaderSource = @R"""(
        #include <metal_stdlib>
        using namespace metal;
        
        // Helper function to compute contained and matching counts
        void check_digits(constant char* answer, constant char* guess, 
                          constant int& n, thread int& contained, thread int& matching) {
            contained = 0;
            matching = 0;
            
            // Arrays to track digit usage (for digits 0-9)
            thread int answer_digits[10] = {0};
            thread int guess_digits[10] = {0};
            
            // Count digits in answer and guess
            for (int i = 0; i < n; i++) {
                answer_digits[answer[i] - '0']++;
                guess_digits[guess[i] - '0']++;
            }
            
            // Calculate contained (total common digits)
            for (int i = 0; i < 10; i++) {
                contained += min(answer_digits[i], guess_digits[i]);
            }
            
            // Calculate matching (correct position)
            for (int i = 0; i < n; i++) {
                if (answer[i] == guess[i]) {
                    matching++;
                }
            }
        }
        
        kernel void calculate_entropies(
            constant char* answer_set,     // All possible answers
            constant char* guesses,        // All guesses to check
            constant int2* pcm,            // All possible (contained, matching) pairs
            device atomic_uint* results,   // Output: count for each guess/pcm combination
            constant uint& answer_count,   // Number of answers
            constant uint& guess_count,    // Number of guesses
            constant uint& pcm_count,      // Number of PCM pairs
            constant int& n,               // Number of digits
            uint3 gid[[thread_position_in_grid]],
            uint3 tid[[thread_position_in_threadgroup]]) {
            
            uint guess_index = gid.x;
            uint answer_index = gid.y;
            
            // Bounds checking
            if (guess_index >= guess_count || answer_index >= answer_count) {
                return;
            }
            
            // Get pointer to the current guess and answer
            constant char* guess = guesses + guess_index * (n + 1);
            constant char* answer = answer_set + answer_index * (n + 1);
            
            // Calculate contained and matching for this guess/answer pair
            int contained, matching;
            check_digits(answer, guess, n, contained, matching);
            
            // Find matching PCM index
            for (uint pcm_index = 0; pcm_index < pcm_count; pcm_index++) {
                if (pcm[pcm_index].x == contained && pcm[pcm_index].y == matching) {
                    // Increment the counter for this guess/pcm combination
                    atomic_fetch_add_explicit(&results[guess_index * pcm_count + pcm_index], 1, memory_order_relaxed);
                    break;
                }
            }
        }
    )""";

    // Compile the shader
    NSError* error = nil;
    pimpl->library = [pimpl->device newLibraryWithSource:shaderSource options:nil error:&error];
    if (error) {
        std::cerr << "MetalSolver::init() - Metal shader compilation error: " << [error.localizedDescription UTF8String] << std::endl;
        return false;
    }
    
    if (!pimpl->library) {
        std::cerr << "MetalSolver::init() - Failed to create Metal library" << std::endl;
        return false;
    }

    // Create the compute pipeline
    id<MTLFunction> kernelFunc = [pimpl->library newFunctionWithName:@"calculate_entropies"];
    if (!kernelFunc) {
        std::cerr << "MetalSolver::init() - Failed to find kernel function" << std::endl;
        return false;
    }
    pimpl->pipelineState = [pimpl->device newComputePipelineStateWithFunction:kernelFunc error:&error];
    if (error) {
        std::cerr << "MetalSolver::init() - Failed to create pipeline state: " << [error.localizedDescription UTF8String] << std::endl;
        return false;
    }
    
    if (!pimpl->pipelineState) {
        std::cerr << "MetalSolver::init() - Failed to create pipeline state" << std::endl;
        return false;
    }
    
    return true;
}

bool MetalSolver::calculate_entropies(
    const std::unordered_set<std::string>& answer_set,
    const std::vector<std::string>& guesses,
    const std::vector<std::pair<int, int>>& pcm,
    std::unordered_map<std::string, std::unordered_map<std::pair<int, int>, int, PairHash>>& results)
{
    if (answer_set.empty() || guesses.empty() || pcm.empty()) {
        std::cerr << "MetalSolver::calculate_entropies() - Empty input data" << std::endl;
        return false;
    }
    
    // Get problem parameters
    int n = static_cast<int>((*answer_set.begin()).length());
    uint answer_count = static_cast<uint>(answer_set.size());
    uint guess_count = static_cast<uint>(guesses.size());
    uint pcm_count = static_cast<uint>(pcm.size());
    
    // Convert answer_set to vector for indexing
    std::vector<std::string> answer_vector(answer_set.begin(), answer_set.end());
    
    // Allocate and fill buffers
    size_t answer_buffer_size = answer_count * (n + 1) * sizeof(char);
    size_t guess_buffer_size = guess_count * (n + 1) * sizeof(char);
    size_t pcm_buffer_size = pcm_count * sizeof(simd_int2);
    size_t results_buffer_size = guess_count * pcm_count * sizeof(uint);
    
    
    // Create Metal buffers
    id<MTLBuffer> answerSetBuffer = [pimpl->device newBufferWithLength:answer_buffer_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> guessesBuffer = [pimpl->device newBufferWithLength:guess_buffer_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> pcmBuffer = [pimpl->device newBufferWithLength:pcm_buffer_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> resultsBuffer = [pimpl->device newBufferWithLength:results_buffer_size options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    
    if (!answerSetBuffer || !guessesBuffer || !pcmBuffer || !resultsBuffer) {
        std::cerr << "MetalSolver::calculate_entropies() - Failed to create Metal buffers" << std::endl;
        return false;
    }
    
    // Fill buffers
    char* answer_data = (char*)answerSetBuffer.contents;
    for (uint i = 0; i < answer_count; i++) {
        memcpy(answer_data + i * (n + 1), answer_vector[i].c_str(), n);
        answer_data[i * (n + 1) + n] = '\0'; // Null terminate
    }
    
    char* guess_data = (char*)guessesBuffer.contents;
    for (uint i = 0; i < guess_count; i++) {
        memcpy(guess_data + i * (n + 1), guesses[i].c_str(), n);
        guess_data[i * (n + 1) + n] = '\0'; // Null terminate
    }
    
    simd_int2* pcm_data = (simd_int2*)pcmBuffer.contents;
    for (uint i = 0; i < pcm_count; i++) {
        pcm_data[i] = simd_make_int2(pcm[i].first, pcm[i].second);
    }
    
    // Initialize results buffer to zero
    memset(resultsBuffer.contents, 0, results_buffer_size);
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [pimpl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    if (!commandBuffer || !computeEncoder) {
        std::cerr << "MetalSolver::calculate_entropies() - Failed to create command buffer or encoder" << std::endl;
        [answerSetBuffer release];
        [guessesBuffer release];
        [pcmBuffer release];
        [resultsBuffer release];
        return false;
    }
    
    // Set the compute pipeline
    [computeEncoder setComputePipelineState:pimpl->pipelineState];
    
    // Bind buffers
    [computeEncoder setBuffer:answerSetBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:guessesBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:pcmBuffer offset:0 atIndex:2];
    [computeEncoder setBuffer:resultsBuffer offset:0 atIndex:3];
    [computeEncoder setBytes:&answer_count length:sizeof(uint) atIndex:4];
    [computeEncoder setBytes:&guess_count length:sizeof(uint) atIndex:5];
    [computeEncoder setBytes:&pcm_count length:sizeof(uint) atIndex:6];
    [computeEncoder setBytes:&n length:sizeof(int) atIndex:7];
    
    // Calculate thread groups with a reasonable size
    // Limit the total number of threads to avoid overwhelming the GPU
    const uint max_threads_per_dimension = 1024;
    uint threadgroups_x = (guess_count + max_threads_per_dimension - 1) / max_threads_per_dimension;
    uint threadgroups_y = (answer_count + max_threads_per_dimension - 1) / max_threads_per_dimension;
    
    // Cap the threadgroups to reasonable values
    threadgroups_x = std::min(threadgroups_x, max_threads_per_dimension);
    threadgroups_y = std::min(threadgroups_y, max_threads_per_dimension);
    
    MTLSize threadsPerThreadgroup = MTLSizeMake(
        std::min(max_threads_per_dimension, guess_count), 
        std::min(max_threads_per_dimension, answer_count), 
        1);
    MTLSize threadgroups = MTLSizeMake(threadgroups_x, threadgroups_y, 1);
    
    // Dispatch the compute kernel
    [computeEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [computeEncoder endEncoding];
    
    // Commit and wait for completion
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Check for errors
    if (commandBuffer.error) {
        std::cerr << "MetalSolver::calculate_entropies() - Command buffer error: " << [commandBuffer.error.localizedDescription UTF8String] << std::endl;
        [answerSetBuffer release];
        [guessesBuffer release];
        [pcmBuffer release];
        [resultsBuffer release];
        return false;
    }
    
    // Process results
    uint* results_data = (uint*)resultsBuffer.contents;
    
    // Convert raw results to the expected format
    for (uint guess_idx = 0; guess_idx < guess_count; guess_idx++) {
        std::string guess = guesses[guess_idx];
        std::unordered_map<std::pair<int, int>, int, PairHash> guess_results;
        
        for (uint pcm_idx = 0; pcm_idx < pcm_count; pcm_idx++) {
            int count = results_data[guess_idx * pcm_count + pcm_idx];
            if (count > 0) {
                guess_results[{pcm[pcm_idx].first, pcm[pcm_idx].second}] = count;
            }
        }
        
        results[guess] = guess_results;
    }
    
    // Release buffers
    [answerSetBuffer release];
    [guessesBuffer release];
    [pcmBuffer release];
    [resultsBuffer release];
    
    return true;
}

std::string MetalSolver::getBackendName() const {
    return "Metal";
}

#endif // __APPLE__