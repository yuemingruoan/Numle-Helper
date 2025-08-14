#ifndef METAL_SOLVER_H
#define METAL_SOLVER_H

#ifdef __APPLE__

#include "IGpuSolver.h"
#include <vector>
#include <string>
#include <memory>

/**
 * @class MetalSolver
 * @brief A Metal-based implementation of the IGpuSolver interface.
 *
 * This class uses the Pimpl idiom to hide Objective-C details from C++ headers,
 * preventing compilation errors in pure C++ files.
 */
class MetalSolver : public IGpuSolver {
public:
    MetalSolver();
    ~MetalSolver() override;

    bool init() override;

    bool calculate_entropies(
        const std::unordered_set<std::string>& answer_set,
        const std::vector<std::string>& guesses,
        const std::vector<std::pair<int, int>>& pcm,
        std::unordered_map<std::string, std::unordered_map<std::pair<int, int>, int, PairHash>>& results) override;

    std::string getBackendName() const override;

private:
    // Forward-declare the implementation class (Pimpl).
    // The actual definition is in the .mm file.
    struct MetalImpl;
    std::unique_ptr<MetalImpl> pimpl;
};

#endif // __APPLE__
#endif // METAL_SOLVER_H
