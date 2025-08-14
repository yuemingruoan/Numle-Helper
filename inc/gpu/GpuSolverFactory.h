#ifndef GPU_SOLVER_FACTORY_H
#define GPU_SOLVER_FACTORY_H

#include "IGpuSolver.h"
#include "CpuSolver.h"
#include "OpenCLSolver.h"
#include "CudaSolver.h"
#ifdef __APPLE__
#include "MetalSolver.h"
#endif

#include <memory>
#include <iostream>

/**
 * @class GpuSolverFactory
 * @brief A factory for creating the best available IGpuSolver instance at runtime.
 *
 * This factory attempts to create solvers in a preferred order (e.g., Metal on Apple,
 * CUDA on NVIDIA) and falls back to more general or CPU-based solvers if the
 * preferred ones are not available or fail to initialize.
 */
class GpuSolverFactory {
public:
    /**
     * @brief Creates and returns the most appropriate solver instance based on runtime availability.
     * @return A unique_ptr to an initialized and ready-to-use IGpuSolver.
     */
    static std::unique_ptr<IGpuSolver> createSolver() {
        std::unique_ptr<IGpuSolver> solver = nullptr;

#ifdef __APPLE__
        // On Apple platforms, try Metal first.
        auto metalSolver = std::make_unique<MetalSolver>();
        if (metalSolver->init()) {
            // Only return the solver without printing message here
            // The message will be printed in XXSNumleSolver::find_best_guess
            return metalSolver;
        }
#else
        // On other platforms (Windows/Linux), try CUDA first.
        auto cudaSolver = std::make_unique<CudaSolver>();
        if (cudaSolver->init()) {
            // Only return the solver without printing message here
            // The message will be printed in XXSNumleSolver::find_best_guess
            return cudaSolver;
        }
#endif

        // If the preferred backend is not available, try OpenCL.
        auto openCLSolver = std::make_unique<OpenCLSolver>();
        if (openCLSolver->init()) {
            // Only return the solver without printing message here
            // The message will be printed in XXSNumleSolver::find_best_guess
            return openCLSolver;
        }

        // If no GPU backends are available, fall back to the CPU solver.
        // The message will be printed in XXSNumleSolver::find_best_guess
        auto cpuSolver = std::make_unique<CpuSolver>();
        // CpuSolver doesn't have an init() method, but we can assume it's always available.
        return cpuSolver;
    }
};

#endif // GPU_SOLVER_FACTORY_H
