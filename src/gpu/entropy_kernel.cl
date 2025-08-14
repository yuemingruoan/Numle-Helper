/**
 * @brief Calculates the 'matching' value between two number strings.
 * 'matching' is the count of digits that are correct in both value and position.
 */
inline int calculate_matching(
    __global const char* answer,
    __global const char* guess,
    const int n_len)
{
    int ret = 0;
    for (int i = 0; i < n_len; ++i) {
        if (answer[i] == guess[i]) {
            ret++;
        }
    }
    return ret;
}

/**
 * @brief Calculates the 'contained' value between two number strings.
 * 'contained' is the count of digits from the answer that are present in the guess,
 * regardless of position.
 */
inline int calculate_contained(
    __global const char* answer,
    __global const char* guess,
    const int n_len)
{
    // Use a bitmask for fast checking of digit presence in the guess.
    // (e.g., '1' -> bit 1, '3' -> bit 3)
    int guess_mask = 0;
    for (int i = 0; i < n_len; ++i) {
        guess_mask |= (1 << (guess[i] - '0'));
    }

    int ret = 0;
    for (int i = 0; i < n_len; ++i) {
        if ((guess_mask & (1 << (answer[i] - '0'))) != 0) {
            ret++;
        }
    }
    return ret;
}

/**
 * @brief Maps a (contained, matching) pair to a unique linear index.
 * This is crucial for placing results correctly in the output buffer.
 * The formula for a full PCM (Possibility Combination Matrix) is:
 * index = (contained * (contained + 1) / 2) + matching
 */
inline int get_pcm_index(int contained, int matching) {
    return (contained * (contained + 1) / 2) + matching;
}

/**
 * @kernel entropy_kernel
 * @brief The main kernel to calculate the distribution of check results.
 *
 * This kernel is launched with a 2D grid of work-items.
 * - get_global_id(0) corresponds to the index of the guess.
 * - get_global_id(1) corresponds to the index of the answer.
 *
 * Each work-item processes one guess-answer pair, calculates the
 * (contained, matching) result, and atomically increments the corresponding
 * counter in the results buffer.
 */
__kernel void entropy_kernel(
    __global const char* answers_buffer, // Flattened char buffer of all answers
    __global const char* guesses_buffer, // Flattened char buffer of all guesses
    __global atomic_int* results_buffer, // Buffer to store the counts for each (guess, pcm) pair
    const int n_len,                     // Length of the number strings (e.g., 4)
    const int pcm_total_size)            // Total number of possible (contained, matching) pairs
{
    int guess_idx = get_global_id(0);
    int answer_idx = get_global_id(1);

    // Pointers to the start of the specific answer and guess strings
    __global const char* current_answer = &answers_buffer[answer_idx * n_len];
    __global const char* current_guess = &guesses_buffer[guess_idx * n_len];

    // Perform the core calculations
    int matching = calculate_matching(current_answer, current_guess, n_len);
    int contained = calculate_contained(current_answer, current_guess, n_len);

    // Map the (contained, matching) result to its linear index
    int pcm_idx = get_pcm_index(contained, matching);

    // Calculate the final position in the results buffer and atomically increment the counter
    // The results are laid out per-guess, so we offset by the guess index.
    int result_idx = guess_idx * pcm_total_size + pcm_idx;
    atomic_inc(&results_buffer[result_idx]);
}
