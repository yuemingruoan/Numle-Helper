#ifndef CPU_SOLVER_H
#define CPU_SOLVER_H

#include "IGpuSolver.h"
#include "XXSNumleChecker.h"

/**
 * @class CpuSolver
 * @brief A CPU-based fallback implementation of the IGpuSolver interface for entropy calculation.
 *
 * This class provides a standard C++ implementation that runs on the CPU.
 * It serves as a baseline and a fallback for when no compatible GPU is available.
 */
class CpuSolver : public IGpuSolver {
public:
    CpuSolver() = default;
    ~CpuSolver() override = default;

    bool init() override;

    bool calculate_entropies(
        const std::unordered_set<std::string>& answer_set,
        const std::vector<std::string>& guesses,
        const std::vector<std::pair<int, int>>& pcm,
        std::unordered_map<std::string, std::unordered_map<std::pair<int, int>, int, PairHash>>& results) override;

    std::string getBackendName() const override;
};

#endif // CPU_SOLVER_H
