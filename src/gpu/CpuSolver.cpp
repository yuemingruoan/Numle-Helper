#include "gpu/CpuSolver.h"
#include <iostream>

bool CpuSolver::init() {
    // CPU solver is always available, so initialization always succeeds.
    return true;
}

bool CpuSolver::calculate_entropies(
    const std::unordered_set<std::string>& answer_set,
    const std::vector<std::string>& guesses,
    const std::vector<std::pair<int, int>>& pcm,
    std::unordered_map<std::string, std::unordered_map<std::pair<int, int>, int, PairHash>>& results)
{
    // This is the original CPU-based logic, now serving as a fallback.
    for (const auto& guess : guesses) {
        std::unordered_map<std::pair<int, int>, int, PairHash> pcm_counts;
        for (const auto& answer : answer_set) {
            int contained = XXSNumleChecker::contained(answer, guess);
            int matching = XXSNumleChecker::matching(answer, guess);
            pcm_counts[{contained, matching}]++;
        }
        results[guess] = pcm_counts;
    }
    return true;
}

std::string CpuSolver::getBackendName() const {
    return "CPU";
}
