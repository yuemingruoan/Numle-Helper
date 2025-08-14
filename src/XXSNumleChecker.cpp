//
// Created by Administrator on 2025/8/12.
//

#include "XXSNumleChecker.h"

XXSNumleChecker::XXSNumleChecker(std::string num)
{
    this->m_num = num;
}

XXSNumleChecker::~XXSNumleChecker() = default;

int XXSNumleChecker::contained(const std::string &guess) const
{
    return XXSNumleChecker::contained(this->m_num, guess);
}

int XXSNumleChecker::matching(const std::string &guess) const
{
    return XXSNumleChecker::matching(this->m_num, guess);
}

XXSNumleChecker::XXSNumleChecker() = default;

void XXSNumleChecker::set_num(const std::string &num)
{
    this->m_num = num;
}

int XXSNumleChecker::contained(const std::string &num, const std::string &guess)
{
    std::unordered_set<char> guess_numbers(guess.begin(), guess.end());
    int ret = 0;
    for(auto &digit : num)
    {
        if(guess_numbers.count(digit))
        {
            ret++;
        }
    }
    return ret;
}

int XXSNumleChecker::matching(const std::string &num, const std::string &guess)
{
    int ret = 0, len = num.length();
    for(int i = 0; i < len; i++)
    {
        if(guess[i] == num[i])
        {
            ret++;
        }
    }
    return ret;
}
