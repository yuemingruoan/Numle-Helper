//
// Created by Administrator on 2025/7/20.
//

#ifndef XXSLIBRARY_XXSSTRINGHELPER_H
#define XXSLIBRARY_XXSSTRINGHELPER_H

#include <string>
#include <sstream>
#include <vector>

/**
 * @brief 字符串助手类, 帮助实现字符串之间的拼接, 宽窄字符串的转换
 */
class XXSStringHelper
{
private:
    std::stringstream  m_ss;
public:
    XXSStringHelper();
    explicit XXSStringHelper(const std::string& str);
    explicit XXSStringHelper(const std::wstring& wstr);
    ~XXSStringHelper();

    std::string from_wstr(const std::wstring& wstr);
    template<typename T>
    XXSStringHelper& read(T& t);
    XXSStringHelper& get_line(std::string& str);
    XXSStringHelper& get_line(std::wstring& wstr);
    template<typename T>
    XXSStringHelper& write(const T& t);
    template<std::size_t N>
    XXSStringHelper& write(const wchar_t(&t)[N]);
    XXSStringHelper& write_line(const std::string& str);
    XXSStringHelper& write_line(const std::wstring& wstr);

    XXSStringHelper& new_line();
    XXSStringHelper& hex();

    XXSStringHelper& erase(int start, int len);
    XXSStringHelper& erase_line(int line);
    XXSStringHelper& insert(int pos, const std::string& str);

    std::string str() const;
    std::wstring wstr() const;

    static std::string cast(const std::wstring& wstr);
    static std::wstring cast(const std::string& str);

    operator bool() const;
};

/**
 * @brief 向字符串中写入数据
 * @tparam T 任何类型
 * @param[in] t 数据
 * @return *this
 */
template<typename T>
XXSStringHelper &XXSStringHelper::write(const T& t)
{
    this->m_ss << t;
    return *this;
}

/**
 * @brief 从字符串中读数据
 * @tparam T 任何类型
 * @param[out] t 用于接受数据的变量引用
 * @return *this
 */
template<typename T>
XXSStringHelper &XXSStringHelper::read(T &t)
{
    this->m_ss >> t;
    return *this;
}

/**
 * @brief 重载 C 风格的宽字符串
 * @tparam N 字符串元素个数
 * @param[in] t 宽字符串
 * @return *this
 */
template<std::size_t N>
XXSStringHelper &XXSStringHelper::write(const wchar_t (&t)[N])
{
    std::wstring wstr = std::wstring(t);
    std::string str = XXSStringHelper::cast(wstr);
    this->m_ss << str;
    return *this;
}

template<>
XXSStringHelper& XXSStringHelper::read<std::wstring>(std::wstring& t);
template<>
XXSStringHelper& XXSStringHelper::write<std::wstring>(const std::wstring& t);
template<>
XXSStringHelper& XXSStringHelper::write<const wchar_t*>(const wchar_t* const &t);

#endif //XXSLIBRARY_XXSSTRINGHELPER_H
