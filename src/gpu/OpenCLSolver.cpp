#include "gpu/OpenCLSolver.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

// Helper function to read kernel file
std::string read_kernel_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open kernel file: " << path << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

OpenCLSolver::OpenCLSolver() {}

OpenCLSolver::~OpenCLSolver() {
    cleanup();
}

void OpenCLSolver::cleanup() {
    if (kernel_) clReleaseKernel(kernel_);
    if (program_) clReleaseProgram(program_);
    if (command_queue_) clReleaseCommandQueue(command_queue_);
    if (context_) clReleaseContext(context_);
}

bool OpenCLSolver::init() {
    cl_int err;

    // Get platform
    err = clGetPlatformIDs(1, &platform_id_, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL Error: Failed to find an OpenCL platform (clGetPlatformIDs failed with code " << err << ")." << std::endl;
        return false;
    }

    // Get platform name for debugging
    char platform_name[256];
    err = clGetPlatformInfo(platform_id_, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    if (err == CL_SUCCESS) {
        std::cout << "[OpenCL] 平台名称: " << platform_name << std::endl;
    }

    // Get device
    err = clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_GPU, 1, &device_id_, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL Info: Failed to find a GPU device (code " << err << "). Trying CPU device..." << std::endl;
        err = clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_CPU, 1, &device_id_, NULL);
        if (err != CL_SUCCESS) {
            std::cerr << "OpenCL Error: Failed to find a CPU device either (clGetDeviceIDs failed with code " << err << ")." << std::endl;
            return false;
        }
    }

    // Get device name for debugging
    char device_name[256];
    err = clGetDeviceInfo(device_id_, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    if (err == CL_SUCCESS) {
        std::cout << "[OpenCL] 设备名称: " << device_name << std::endl;
    }

    // Create context
    context_ = clCreateContext(0, 1, &device_id_, NULL, NULL, &err);
    if (!context_ || err != CL_SUCCESS) {
        std::cerr << "OpenCL Error: Failed to create a compute context (clCreateContext failed with code " << err << ")." << std::endl;
        return false;
    }

    // Create command queue
    command_queue_ = clCreateCommandQueue(context_, device_id_, 0, &err);
    if (!command_queue_ || err != CL_SUCCESS) {
        std::cerr << "OpenCL Error: Failed to create a command queue (clCreateCommandQueue failed with code " << err << ")." << std::endl;
        return false;
    }

    // Create program from source - try both relative and absolute paths for the kernel file
    std::string kernel_source = read_kernel_file("entropy_kernel.cl");
    if (kernel_source.empty()) {
        // Try to find the kernel file in current directory or build directory
        const char* possible_paths[] = {"./entropy_kernel.cl", "../src/gpu/entropy_kernel.cl", "src/gpu/entropy_kernel.cl"};
        for (const auto& path : possible_paths) {
            kernel_source = read_kernel_file(path);
            if (!kernel_source.empty()) {
                std::cout << "[OpenCL] 在路径 '" << path << "' 找到内核文件" << std::endl;
                break;
            }
        }
    }
    if (kernel_source.empty()) {
        std::cerr << "OpenCL Error: Kernel source file is empty or could not be read." << std::endl;
        return false;
    }
    const char* source_str = kernel_source.c_str();
    size_t source_size = kernel_source.length();

    program_ = clCreateProgramWithSource(context_, 1, &source_str, &source_size, &err);
    if (!program_ || err != CL_SUCCESS) {
        std::cerr << "OpenCL Error: Failed to create compute program (clCreateProgramWithSource failed with code " << err << ")." << std::endl;
        return false;
    }

    // Build program
    err = clBuildProgram(program_, 1, &device_id_, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[4096];
        std::cerr << "OpenCL Error: Failed to build program executable (clBuildProgram failed with code " << err << ")." << std::endl;
        clGetProgramBuildInfo(program_, device_id_, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        std::cerr << "--- OpenCL Build Log ---" << std::endl << buffer << std::endl;
        return false;
    }

    // Create kernel
    kernel_ = clCreateKernel(program_, "entropy_kernel", &err);
    if (!kernel_ || err != CL_SUCCESS) {
        std::cerr << "OpenCL Error: Failed to create compute kernel (clCreateKernel failed with code " << err << ")." << std::endl;
        return false;
    }
    
    std::cout << "[OpenCL] OpenCL后端初始化成功" << std::endl;
    return true;
}

bool OpenCLSolver::calculate_entropies(
    const std::unordered_set<std::string>& answer_set,
    const std::vector<std::string>& guesses,
    const std::vector<std::pair<int, int>>& pcm,
    std::unordered_map<std::string, std::unordered_map<std::pair<int, int>, int, PairHash>>& results)
{
    if (answer_set.empty() || guesses.empty()) {
        return true;
    }

    cl_int err;
    const int n_len = guesses[0].length();
    const int pcm_total_size = (n_len + 1) * (n_len + 2) / 2;

    // 1. Flatten data for GPU
    std::vector<char> answers_flat;
    answers_flat.reserve(answer_set.size() * n_len);
    for (const auto& s : answer_set) {
        answers_flat.insert(answers_flat.end(), s.begin(), s.end());
    }

    std::vector<char> guesses_flat;
    guesses_flat.reserve(guesses.size() * n_len);
    for (const auto& s : guesses) {
        guesses_flat.insert(guesses_flat.end(), s.begin(), s.end());
    }

    // 2. Create GPU buffers
    size_t results_buffer_size = sizeof(cl_int) * guesses.size() * pcm_total_size;
    std::vector<cl_int> host_results(guesses.size() * pcm_total_size, 0);

    cl_mem answers_buf = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, answers_flat.size() * sizeof(char), answers_flat.data(), &err);
    if (err != CL_SUCCESS) { 
        std::cerr << "OpenCL Error: Failed to create answers buffer (code " << err << ")." << std::endl; 
        return false; 
    }
    
    cl_mem guesses_buf = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, guesses_flat.size() * sizeof(char), guesses_flat.data(), &err);
    if (err != CL_SUCCESS) { 
        std::cerr << "OpenCL Error: Failed to create guesses buffer (code " << err << ")." << std::endl; 
        clReleaseMemObject(answers_buf); 
        return false; 
    }

    cl_mem results_buf = clCreateBuffer(context_, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, results_buffer_size, host_results.data(), &err);
    if (err != CL_SUCCESS) { 
        std::cerr << "OpenCL Error: Failed to create results buffer (code " << err << ")." << std::endl; 
        clReleaseMemObject(answers_buf); 
        clReleaseMemObject(guesses_buf); 
        return false; 
    }

    // 3. Set kernel arguments
    err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &answers_buf);
    err |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &guesses_buf);
    err |= clSetKernelArg(kernel_, 2, sizeof(cl_mem), &results_buf);
    err |= clSetKernelArg(kernel_, 3, sizeof(cl_int), &n_len);
    err |= clSetKernelArg(kernel_, 4, sizeof(cl_int), &pcm_total_size);
    if (err != CL_SUCCESS) { 
        std::cerr << "OpenCL Error: Failed to set kernel arguments (code " << err << ")." << std::endl; 
        clReleaseMemObject(answers_buf); 
        clReleaseMemObject(guesses_buf); 
        clReleaseMemObject(results_buf); 
        return false; 
    }

    // 4. Execute kernel
    size_t global_work_size[2] = {guesses.size(), answer_set.size()};
    err = clEnqueueNDRangeKernel(command_queue_, kernel_, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { 
        std::cerr << "OpenCL Error: Failed to execute kernel (clEnqueueNDRangeKernel failed with code " << err << ")." << std::endl; 
        clReleaseMemObject(answers_buf); 
        clReleaseMemObject(guesses_buf); 
        clReleaseMemObject(results_buf); 
        return false; 
    }

    // 5. Read back results
    err = clEnqueueReadBuffer(command_queue_, results_buf, CL_TRUE, 0, results_buffer_size, host_results.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) { 
        std::cerr << "OpenCL Error: Failed to read output buffer (clEnqueueReadBuffer failed with code " << err << ")." << std::endl; 
        clReleaseMemObject(answers_buf); 
        clReleaseMemObject(guesses_buf); 
        clReleaseMemObject(results_buf); 
        return false; 
    }

    // 6. Unpack results from flat buffer into map
    for (size_t i = 0; i < guesses.size(); ++i) {
        std::unordered_map<std::pair<int, int>, int, PairHash> pcm_counts;
        for (const auto& p : pcm) {
            int contained = p.first;
            int matching = p.second;
            int pcm_idx = (contained * (contained + 1) / 2) + matching;
            int count = host_results[i * pcm_total_size + pcm_idx];
            if (count > 0) {
                pcm_counts[p] = count;
            }
        }
        results[guesses[i]] = pcm_counts;
    }

    // 7. Cleanup
    clReleaseMemObject(answers_buf);
    clReleaseMemObject(guesses_buf);
    clReleaseMemObject(results_buf);

    return true;
}

std::string OpenCLSolver::getBackendName() const {
    return "OpenCL";
}