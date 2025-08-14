//
// Created by Administrator on 2025/7/20.
//

#include "XXSDefines.h"
#include "XXSConsole.h"

#if defined(XXS_PLATFORM_WINDOWS)
#include <windows.h>
#include <io.h>
#endif

int XXSConsole::m_counter = 0;
bool XXSConsole::m_b_enabled_ansi_esc_codes = false;

/**
 * @brief 初始化工作
 */
XXSConsole::XXSConsole()
{
#if defined(XXS_PLATFORM_WINDOWS)
    //首次实例化
    if(!XXSConsole::m_counter)
    {
        // 设置 stdin 和 stdout 的模式
        this->m_old_stdin = _setmode(_fileno(stdin), _O_U16TEXT);
        this->m_old_stdout = _setmode(_fileno(stdout), _O_BINARY);
    }
#endif
    XXSConsole::m_counter++;
}

/**
 * @brief 还原原来的文件模式
 */
XXSConsole::~XXSConsole()
{
    XXSConsole::m_counter--;
#if defined(XXS_PLATFORM_WINDOWS)
    if(!XXSConsole::m_counter)
    {
        _setmode(_fileno(stdin), this->m_old_stdin);
        _setmode(_fileno(stdout), this->m_old_stdout);
    }
#endif
}

/**
 * @brief 从控制台读取一行的数据
 * @param[out] str 接受字符串的引用
 * @return *this
 */
XXSConsole &XXSConsole::get_line(std::string &str)
{
    std::wstring wstr;
    std::getline(std::wcin, wstr);
    str = XXSStringHelper::cast(wstr);
    return *this;
}

/**
 * @brief 从控制台读取一行的数据到宽字符串
 * @param[out] wstr 接受宽字符串的引用
 * @return *this
 */
XXSConsole &XXSConsole::get_line(std::wstring &wstr)
{
    std::getline(std::wcin, wstr);
    return *this;
}

/**
 * @brief 向控制台写入一行的数据
 * @param[in] str 数据
 * @return *this
 */
XXSConsole &XXSConsole::write_line(const std::string &str)
{
    std::cout << str << std::endl;
    return *this;
}

/**
 * @brief 向控制台写入一行数据
 * @param[in] wstr 宽字符串
 * @return *this
 */
XXSConsole &XXSConsole::write_line(const std::wstring &wstr)
{
    std::string str = XXSStringHelper::cast(wstr);
    std::cout << str << std::endl;
    return *this;
}

/**
 * @brief 初始化 UTF-8 控制台, 解决中文乱码问题
 */
void XXSConsole::init_encoding()
{
    // 设置 locale 和 控制台的 CodePage
    std::setlocale(LC_ALL, "zh_CN.UTF-8");
#if defined(XXS_PLATFORM_WINDOWS)
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
}

/**
 * @brief 激活 ANSI 转义序列
 * @param[in] b_enable 是否激活
 */
void XXSConsole::enable_ansi_esc_codes(bool b_enable)
{
    XXSConsole::m_b_enabled_ansi_esc_codes = b_enable;
#if defined(XXS_PLATFORM_WINDOWS)
    unsigned long mode;
    HANDLE h_console = GetStdHandle(STD_OUTPUT_HANDLE);
    GetConsoleMode(h_console, &mode);
    mode &= ~ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(h_console, b_enable ? mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING : mode);
#endif
}

/**
 * @brief 设置控制台文本的颜色
 * @param[in] r 颜色的 r 分量
 * @param[in] g 颜色的 g 分量
 * @param[in] b 颜色的 b 分量
 * @return *this
 */
XXSConsole &XXSConsole::set_text_color(int r, int g, int b)
{
    if(XXSConsole::m_b_enabled_ansi_esc_codes)
    {
        XXSStringHelper sh;
        sh.write("\033[38;2;").write(r).write(";").write(g).write(";").write(b).write("m");
        std::cout << sh.str();
    }
    return *this;
}

/**
 * @brief 设置控制台文本背景的颜色
 * @param[in] r 颜色的 r 分量
 * @param[in] g 颜色的 g 分量
 * @param[in] b 颜色的 b 分量
 * @return *this
 */
XXSConsole &XXSConsole::set_background_color(int r, int g, int b)
{
    if(XXSConsole::m_b_enabled_ansi_esc_codes)
    {
        XXSStringHelper sh;
        sh.write("\033[48;2;").write(r).write(";").write(g).write(";").write(b).write("m");
        std::cout << sh.str();
    }
    return *this;
}

/**
 * @brief 重设控制台文本的颜色
 * @return *this
 */
XXSConsole &XXSConsole::reset_color()
{
    if(XXSConsole::m_b_enabled_ansi_esc_codes)
    {
        std::cout << "\033[0m";
    }
    return *this;
}

/**
 * @brief 设置控制台文本的颜色
 * @param[in] rgb 一个 RGB 的十六进制码, 比如 0xffffff
 * @return *this
 */
XXSConsole &XXSConsole::set_text_color(long long int rgb)
{
    if(XXSConsole::m_b_enabled_ansi_esc_codes)
    {
        // 位运算提取 r, g, b 分量
        int r = (rgb >> 16) & 0xff;
        int g = (rgb >> 8) & 0xff;
        int b = rgb & 0xff;
        this->set_text_color(r, g, b);
    }
    return *this;
}

/**
 * @brief 设置控制台文本背景的颜色
 * @param[in] rgb 一个 RGB 的十六进制码, 比如 0xffffff
 * @return *this
 */
XXSConsole &XXSConsole::set_background_color(long long int rgb)
{
    if(XXSConsole::m_b_enabled_ansi_esc_codes)
    {
        // 位运算提取 r, g, b 分量
        int r = (rgb >> 16) & 0xff;
        int g = (rgb >> 8) & 0xff;
        int b = rgb & 0xff;
        this->set_background_color(r, g, b);
    }
    return *this;
}

/**
 * @brief 后一个输出的数字将以十六进制输出 (不带 0x )
 * @return *this
 */
XXSConsole &XXSConsole::hex()
{
    std::cout << std::hex;
    return *this;
}

/**
 * @brief 换行
 * @return *this
 */
XXSConsole &XXSConsole::new_line()
{
    std::cout << std::endl;
    return *this;
}

/**
 * @brief 从控制台读取窄字符串
 * @param[out] t 接受窄字符串的变量引用
 * @return *this
 */
template<>
XXSConsole &XXSConsole::read<std::string>(std::string &t)
{
    std::wstring wstr;
    std::wcin >> wstr;
    t = XXSStringHelper::cast(wstr);
    return *this;
}

/**
 * @brief 向控制台写入宽字符串
 * @param[in] t 宽字符串
 * @return *this
 */
template<>
XXSConsole &XXSConsole::write<std::wstring>(const std::wstring &t)
{
    std::string str = XXSStringHelper::cast(t);
    std::cout << str;
    return *this;
}

/**
 * @brief 向控制台写入宽字符串
 * @param[in] t 宽字符串
 * @return *this
 */
template<>
XXSConsole &XXSConsole::write<const wchar_t*>(const wchar_t* const &t)
{
    std::string str = XXSStringHelper::cast(t);
    std::cout << str;
    return *this;
}
