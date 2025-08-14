//
// Created by Administrator on 2025/7/20.
//

#include "XXSStringHelper.h"

#if !defined(XXS_PLATFORM_WINDOWS)
#include <codecvt>
#include <locale>
#endif

/**
 * @brief 初始化字符串流为空
 */
XXSStringHelper::XXSStringHelper()
{
    this->m_ss.str("");
}

/**
 * @brief 用窄字符串来初始化
 * @param[in] str 窄字符串
 */
XXSStringHelper::XXSStringHelper(const std::string& str)
{
    this->m_ss.str(str);
}

/**
 * @brief 用宽字符串来初始化
 * @param[in] wstr 宽字符串
 */
XXSStringHelper::XXSStringHelper(const std::wstring& wstr)
{
    this->from_wstr(wstr);
}

/**
 * @brief 用宽字符串来初始化
 * @param[in] wstr 宽字符串
 * @return 转换后的窄字符串
 */
std::string XXSStringHelper::from_wstr(const std::wstring &wstr)
{
    std::string str_ret = XXSStringHelper::cast(wstr);
    this->m_ss = std::stringstream(str_ret);
    return str_ret;
}

/**
 * @brief 获取一行的内容
 * @param[out] str 接受内容的字符串引用
 * @return *this
 */
XXSStringHelper &XXSStringHelper::get_line(std::string &str)
{
    getline(this->m_ss, str);
    return *this;
}

/**
 * @brief 向字符串写入一行数据
 * @param[in] str 字符串内容
 * @return *this
 */
XXSStringHelper &XXSStringHelper::write_line(const std::string &str)
{
    this->m_ss << str << std::endl;
    return *this;
}

/**
 * @brief 实现宽字符串和窄字符串之间的转换
 * @param[in] wstr 宽字符串
 * @return 窄字符串
 */
std::string XXSStringHelper::cast(const std::wstring& wstr)
{
#if defined(XXS_PLATFORM_WINDOWS)
    std::string str_ret;
    int len = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), static_cast<int>(wstr.size()),nullptr, 0, nullptr, nullptr);
    auto buf = new char[len + 1];
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), static_cast<int>(wstr.size()), buf, len, nullptr, nullptr);
    buf[len] = 0;
    str_ret = buf;
    delete[] buf;
    return str_ret;
#else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    return conv.to_bytes(wstr);
#endif
}

/**
 * @brief 实现宽字符串和窄字符串之间的转换
 * @param[in] str 窄字符串
 * @return 宽字符串
 */
std::wstring XXSStringHelper::cast(const std::string& str)
{
#if defined(XXS_PLATFORM_WINDOWS)
    std::wstring wstr_ret;
    int len = MultiByteToWideChar(CP_UTF8, 0, str.c_c_str(), static_cast<int>(str.size()), nullptr, 0);
    auto buf = new wchar_t[len + 1];
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), buf, len);
    buf[len] = 0;
    wstr_ret = buf;
    delete[] buf;
    return wstr_ret;
#else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    return conv.from_bytes(str);
#endif
}

/**
 * @brief 读取一行数据到宽字符串
 * @param[out] wstr 接受宽字符串的引用
 * @return *this
 */
XXSStringHelper &XXSStringHelper::get_line(std::wstring &wstr)
{
    std::string str;
    this->get_line(str);
    wstr = XXSStringHelper::cast(str);
    return *this;
}

/**
 * @brief 向字符串中写入一行宽字符串
 * @param[in] wstr 宽字符串
 * @return *this
 */
XXSStringHelper &XXSStringHelper::write_line(const std::wstring &wstr)
{
    std::string str = XXSStringHelper::cast(wstr);
    this->write_line(str);
    return *this;
}

/**
 * @brief 获得字符串
 * @return 字符串
 */
std::string XXSStringHelper::str() const
{
    return this->m_ss.str();
}

/**
 * @brief 获得宽字符串
 * @return 宽字符串
 */
std::wstring XXSStringHelper::wstr() const
{
    return XXSStringHelper::cast(this->m_ss.str());
}

/**
 * @brief 检查上一个操作是否合法
 * @return 操作是否合法
 */
XXSStringHelper::operator bool() const
{
    return !this->m_ss.fail();
}

/**
 * @brief 擦除字符串的一段数据
 * @param[in] start 起始位置
 * @param[in] len 长度
 * @return *this
 */
XXSStringHelper &XXSStringHelper::erase(int start, int len)
{
    std::string str = this->m_ss.str();
    str.erase(start, len);
    this->m_ss = std::stringstream(str);
    return *this;
}

/**
 * @brief 删除字符串的某一行
 * @param[in] line 行号,从 1 开始编号
 * @return *this
 */
XXSStringHelper &XXSStringHelper::erase_line(int line)
{
    if(line <= 0)
    {
        return *this;
    }

    std::string str = this->m_ss.str();
    std::stringstream ss(str);
    std::string str_line;
    int start = 0, i = line;
    while(std::getline(ss, str_line))
    {
        i--;
        int len = str_line.length() + ((str[start + str_line.length()] == '\0') ? 0 : 1);
        if(i <= 0)
        {
            str.erase(start, len);
            break;
        }
        start+=len;
    }
    this->m_ss = std::stringstream(str);
    return *this;
}

/**
 * @brief 在字符串的某个位置插入一串字符串
 * @param[in] pos 位置
 * @param[in] str 字符串
 * @return *this
 */
XXSStringHelper &XXSStringHelper::insert(int pos, const std::string &str)
{
    std::string str0 = this->m_ss.str();
    str0.insert(pos, str);
    this->m_ss = std::stringstream(str0);
    return *this;
}

/**
 * @brief 换行
 * @return *this
 */
XXSStringHelper &XXSStringHelper::new_line()
{
    this->m_ss << std::endl;
    return *this;
}

/**
 * @brief 下一个输出的数字将以十六进制输出 ( 不带 0x )
 * @return *this
 */
XXSStringHelper &XXSStringHelper::hex()
{
    this->m_ss << std::hex;
    return *this;
}

/**
 * @brief 特化宽字符串的情况, 转换一下再写入字符串
 * @param[in] t 宽字符串
 * @return *this
 */
template<>
XXSStringHelper &XXSStringHelper::write<std::wstring>(const std::wstring &t)
{
    std::string str = XXSStringHelper::cast(t);
    this->m_ss << str;
    return *this;
}

/**
 * @brief 特化宽字符串的情况, 自带一个转换
 * @param[out] t
 * @return *this
 */
template<>
XXSStringHelper &XXSStringHelper::read<std::wstring>(std::wstring &t)
{
    std::string str;
    this->m_ss >> str;
    t = XXSStringHelper::cast(str);
    return *this;
}

/**
 * @brief 特化宽字符串的情况, 转换一下再写入字符串
 * @param[in] t 宽字符串
 * @return *this
 */
template<>
XXSStringHelper& XXSStringHelper::write<const wchar_t*>(const wchar_t* const &t)
{
    std::string str = XXSStringHelper::cast(t);
    this->m_ss << str;
    return *this;
}

XXSStringHelper::~XXSStringHelper() = default;
