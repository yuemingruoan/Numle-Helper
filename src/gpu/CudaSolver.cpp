#include "gpu/CudaSolver.h"
#include <iostream>

// 尝试包含CUDA头文件
#ifdef __has_include
#  if __has_include(<cuda.h>) && __has_include(<cuda_runtime.h>)
#    include <cuda.h>
#    include <cuda_runtime.h>
#    define CUDA_AVAILABLE
#  endif
#endif

CudaSolver::CudaSolver() {}

CudaSolver::~CudaSolver() {}

bool CudaSolver::init() {
#ifdef CUDA_AVAILABLE
    // 检查CUDA设备是否可用
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "[CUDA] CUDA初始化失败: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    if (deviceCount == 0) {
        std::cerr << "[CUDA] 未找到可用的CUDA设备" << std::endl;
        return false;
    }
    
    // 尝试初始化第一个设备
    cudaDeviceProp deviceProp;
    error = cudaGetDeviceProperties(&deviceProp, 0);
    if (error != cudaSuccess) {
        std::cerr << "[CUDA] 获取设备属性失败: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    std::cout << "[CUDA] 找到设备: " << deviceProp.name << std::endl;
    return true;
#else
    // CUDA不可用
    return false;
#endif
}

bool CudaSolver::calculate_entropies(
    const std::unordered_set<std::string>& answer_set,
    const std::vector<std::string>& guesses,
    const std::vector<std::pair<int, int>>& pcm,
    std::unordered_map<std::string, std::unordered_map<std::pair<int, int>, int, PairHash>>& results)
{
#ifdef CUDA_AVAILABLE
    // 这里应该实现真正的CUDA计算逻辑
    // 由于这是一个复杂的实现，暂时返回false以回退到CPU版本
    // 实际项目中应该实现完整的CUDA内核和数据传输逻辑
    std::cerr << "[CUDA] CUDA计算功能尚未实现，回退到CPU计算" << std::endl;
    return false;
#else
    std::cerr << "[CUDA] CUDA不可用，回退到CPU计算" << std::endl;
    return false;
#endif
}

std::string CudaSolver::getBackendName() const {
    return "CUDA";
}
