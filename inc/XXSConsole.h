//
// Created by Administrator on 2025/7/20.
//

#ifndef XXSLIBRARY_XXSCONSOLE_H
#define XXSLIBRARY_XXSCONSOLE_H

#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <locale>
#include "XXSStringHelper.h"

/**
 * @brief 控制台类, 用来在控制台输入和输出
 */
class XXSConsole
{
private:
    static int m_counter;
    static bool m_b_enabled_ansi_esc_codes;
    int m_old_stdin;
    int m_old_stdout;
public:
    XXSConsole();
    ~XXSConsole();

    static void init_encoding();
    static void enable_ansi_esc_codes(bool b_enable);

    template<typename T>
    XXSConsole& read(T& t);
    XXSConsole& get_line(std::string& str);
    XXSConsole& get_line(std::wstring& wstr);
    template<typename T>
    XXSConsole& write(const T& t);
    template<std::size_t N>
    XXSConsole& write(const wchar_t(&t)[N]);
    XXSConsole& write_line(const std::string& str);
    XXSConsole& write_line(const std::wstring& wstr);

    XXSConsole& new_line();
    XXSConsole& hex();

    // ANSI 转义序列相关

    XXSConsole& set_text_color(int r, int g, int b);
    XXSConsole& set_text_color(long long rgb);
    XXSConsole& set_background_color(int r, int g, int b);
    XXSConsole& set_background_color(long long rgb);
    XXSConsole& reset_color();
};

/**
 * @brief 从控制台读入一个数据
 * @tparam T 任何类型
 * @param[out] t 接受数据的变量引用
 * @return *this
 */
template<typename T>
XXSConsole &XXSConsole::read(T &t)
{
    std::wcin >> t;
    return *this;
}

/**
 * @brief 向控制台写入一个数据
 * @tparam T 任何类型
 * @param[in] t 数据
 * @return *this
 */
template<typename T>
XXSConsole &XXSConsole::write(const T &t)
{
    std::cout << t;
    return *this;
}

/**
 * @brief 输出宽字符串
 * @tparam N 字符串元素个数
 * @param[in] t 宽字符串
 * @return *this
 */
template<std::size_t N>
XXSConsole &XXSConsole::write(const wchar_t (&t)[N])
{
    std::wstring wstr(t);
    std::cout << XXSStringHelper::cast(wstr);
    return *this;
}

template<>
XXSConsole &XXSConsole::read<std::string>(std::string &t);
template<>
XXSConsole &XXSConsole::write<std::wstring>(const std::wstring &t);
template<>
XXSConsole &XXSConsole::write<const wchar_t*>(const wchar_t* const &t);

#endif //XXSLIBRARY_XXSCONSOLE_H
