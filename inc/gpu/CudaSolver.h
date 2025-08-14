#ifndef CUDA_SOLVER_H
#define CUDA_SOLVER_H

#include "IGpuSolver.h"
#include <vector>
#include <string>

/**
 * @class CudaSolver
 * @brief A CUDA-based implementation of the IGpuSolver interface.
 *
 * This class is intended to run on NVIDIA GPUs using the CUDA toolkit.
 * Note: This is a stub implementation.
 */
class CudaSolver : public IGpuSolver {
public:
    CudaSolver();
    ~CudaSolver() override;

    bool init() override;

    bool calculate_entropies(
        const std::unordered_set<std::string>& answer_set,
        const std::vector<std::string>& guesses,
        const std::vector<std::pair<int, int>>& pcm,
        std::unordered_map<std::string, std::unordered_map<std::pair<int, int>, int, PairHash>>& results) override;

    std::string getBackendName() const override;
};

#endif // CUDA_SOLVER_H
