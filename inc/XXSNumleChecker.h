//
// Created by Administrator on 2025/8/12.
//

#ifndef XXSLIBRARY_XXSNUMLECHECKER_H
#define XXSLIBRARY_XXSNUMLECHECKER_H

#include <string>
#include <unordered_set>

/**
 * @brief Numle 比对器, 给一个数字, 去判断有几个包含, 几个位置正确
 */
class XXSNumleChecker
{
public:
    XXSNumleChecker();
    XXSNumleChecker(std::string num);
    ~XXSNumleChecker();

    void set_num(const std::string &num);

    int contained(const std::string &guess) const;
    int matching(const std::string &guess) const;

    static int contained(const std::string &num, const std::string &guess);
    static int matching(const std::string &num, const std::string &guess);
private:
    std::string m_num;
};


#endif //XXSLIBRARY_XXSNUMLECHECKER_H
