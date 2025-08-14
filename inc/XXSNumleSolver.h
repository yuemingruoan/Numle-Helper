//
// Created by Administrator on 2025/8/12.
//

#ifndef XXSLIBRARY_XXSNUMLESOLVER_H
#define XXSLIBRARY_XXSNUMLESOLVER_H

#include "XXSNumleChecker.h"
#include "XXSConsole.h"
#include "XXSFile.h"
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <map>
#include <set>
#include <queue>
#include <thread>
#include <mutex>
#include <vector>

template <class T>
inline void _xxs_hash_conbine(std::size_t& seed, const T& val)
{
    seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct _xxs_tmp_hash1{
    std::size_t operator()(const std::unordered_map<std::string, std::pair<int, int>>& inner_map) const
    {
        std::size_t seed = 0;
        for (const auto& entry : inner_map)
        {
            std::size_t sub_seed = 0;
            _xxs_hash_conbine(sub_seed, entry.first);
            _xxs_hash_conbine(sub_seed, entry.second.first);
            _xxs_hash_conbine(sub_seed, entry.second.second);
            seed += sub_seed;
        }
        return seed;
    }
};

class XXSNumleSolver
{
public:
    XXSNumleSolver();
    XXSNumleSolver(int n);
    ~XXSNumleSolver();

    void set_digit(int n);
    void init_set();
    long double restrict(const std::string &guess, int contained, int matching);

    int get_set_size() const;
    const std::unordered_set<std::string> &get_set() const;
    long double get_guess_entropy(const std::string &guess) const;
    std::pair<std::string, long double> get_best_guess(bool b_show_progress = false) const;
    const std::unordered_map<
        std::unordered_map<
            std::string,
            std::pair<int, int>>,
        std::vector<std::pair<
            std::string,
            long double>>,
        _xxs_tmp_hash1> &get_table() const;

    void make_table(const std::string &filename);
    void read_table(const std::string &filename);

    static std::unordered_set<std::string> get_permutations(int n);
    static void restrict_set(std::unordered_set<std::string> &set, const std::string &guess, int contained, int matching);
    static long double calc_entropy(const std::unordered_set<std::string> &set, const std::vector<std::pair<int, int>> &pcm, const std::string &guess);
    static long double calc_possibility(const std::unordered_set<std::string> &set, const std::string &guess, int contained, int matching);
    static std::unordered_map<std::string, long double> calc_guess_entropy_map(const std::unordered_set<std::string> &set, const std::vector<std::pair<int, int>> &pcm, const std::unordered_set<std::string> &guesses, bool b_show_progress = false);
    static std::pair<std::string, long double> find_best_guess(const std::unordered_set<std::string> &set, const std::vector<std::pair<int, int>> &pcm, const std::unordered_set<std::string> &guesses, bool b_show_progress = false);
private:
    int m_n;
    std::unordered_set<std::string> m_set;
    std::unordered_set<std::string> m_guesses;
    std::vector<std::pair<int, int>> m_pcm;
    std::unordered_map<
        std::string,
        std::pair<int, int>> m_history;
    std::unordered_map<
        std::unordered_map<
            std::string,
            std::pair<int, int>>,
        std::vector<std::pair<
            std::string,
            long double>>,
        _xxs_tmp_hash1> m_M;
};


#endif //XXSLIBRARY_XXSNUMLESOLVER_H
