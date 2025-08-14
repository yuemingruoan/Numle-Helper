#ifndef OPENCL_SOLVER_H
#define OPENCL_SOLVER_H

#include "IGpuSolver.h"

// Handle different locations for OpenCL headers
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <vector>
#include <string>

/**
 * @class OpenCLSolver
 * @brief An OpenCL-based implementation of the IGpuSolver interface.
 *
 * This class uses the OpenCL framework to perform computations on a compatible GPU.
 * It handles the setup of the OpenCL environment (platform, device, context, queue),
 * kernel compilation, and execution.
 */
class OpenCLSolver : public IGpuSolver {
public:
    OpenCLSolver();
    ~OpenCLSolver() override;

    bool init() override;

    bool calculate_entropies(
        const std::unordered_set<std::string>& answer_set,
        const std::vector<std::string>& guesses,
        const std::vector<std::pair<int, int>>& pcm,
        std::unordered_map<std::string, std::unordered_map<std::pair<int, int>, int, PairHash>>& results) override;

    std::string getBackendName() const override;

private:
    cl_platform_id platform_id_ = nullptr;
    cl_device_id device_id_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue command_queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;

    /**
     * @brief Cleans up all allocated OpenCL resources.
     */
    void cleanup();
};

#endif // OPENCL_SOLVER_H
