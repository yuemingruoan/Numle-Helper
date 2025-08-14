#ifndef I_GPU_SOLVER_H
#define I_GPU_SOLVER_H

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <utility>

/**
 * @class IGpuSolver
 * @brief An interface for a generic GPU-accelerated entropy calculator.
 *
 * This class defines the contract for backends (CUDA, Metal, OpenCL) that calculate
 * the distribution of check results (contained, matching) for a list of guesses
 * against a set of possible answers. This distribution is the core component needed
 * to calculate information entropy.
 */
class IGpuSolver {
public:
    // Hash function for std::pair to be used in unordered_map
    struct PairHash {
        template <class T1, class T2>
        std::size_t operator() (const std::pair<T1, T2>& p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            // A simple way to combine hashes
            return h1 ^ (h2 << 1);
        }
    };

    virtual ~IGpuSolver() = default;

    /**
     * @brief Initializes the GPU backend.
     * @return True if initialization was successful, false otherwise.
     */
    virtual bool init() = 0;

    /**
     * @brief Calculates the result distribution for a batch of guesses against a set of answers.
     *
     * For each guess, this function computes how many answers in the set produce each
     * possible {contained, matching} pair.
     *
     * @param answer_set The current set of all possible answers.
     * @param guesses The list of guesses to evaluate.
     * @param pcm The list of all possible {contained, matching} pairs.
     * @param results A map where the key is the guess and the value is another map.
     *                The inner map's key is the {contained, matching} pair, and its value
     *                is the count of answers that produce this result for the given guess.
     * @return True if the computation was successful, false otherwise.
     */
    virtual bool calculate_entropies(
        const std::unordered_set<std::string>& answer_set,
        const std::vector<std::string>& guesses,
        const std::vector<std::pair<int, int>>& pcm,
        std::unordered_map<std::string, std::unordered_map<std::pair<int, int>, int, PairHash>>& results) = 0;

    /**
     * @brief Returns the name of the backend implementation.
     * @return A string identifying the backend (e.g., "CUDA", "Metal", "OpenCL", "CPU").
     */
    virtual std::string getBackendName() const = 0;
};

#endif // I_GPU_SOLVER_H
