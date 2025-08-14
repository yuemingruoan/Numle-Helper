//
// Created by Administrator on 2025/7/21.
//

#include "XXSFile.h"

XXSFile::XXSFile() = default;

XXSFile::~XXSFile() = default;

/**
 * @brief 实例化的时候设置文件名
 * @param[in] filename 文件名
 */
XXSFile::XXSFile(const std::string &filename)
{
    this->m_filename = filename;
}

/**
 * @brief 设置文件名
 * @param[in] filename 文件名
 */
void XXSFile::set_filename(const std::string &filename)
{
    this->m_filename = filename;
}

/**
 * @brief 设置文件的数据(如果有一次性覆盖写的需求)
 * @param[in] data 数据
 */
void XXSFile::set_data(const std::string &data)
{
    this->m_data = data;
}

/**
 * @brief 获得文件名字(用别的接口传回文件实例时)
 * @return 文件的名字
 */
const std::string &XXSFile::get_filename() const
{
    return this->m_filename;
}

/**
 * @brief 获得文件全部数据(如果有一次性读完的需求)
 * @return 数据
 */
const std::string &XXSFile::get_data() const
{
    return this->m_data;
}

/**
 * @brief 加载文件数据进 data
 * @return 是否成功加载文件
 */
bool XXSFile::load()
{
    std::fstream fs;
    fs.open(this->m_filename, std::ios::in | std::ios::binary);
    if (!fs.is_open()) {
        return false;
    }
    
    XXSStringHelper sh;
    std::string str_line;
    while(std::getline(fs, str_line))
    {
        sh.write_line(str_line);
    }
    fs.close();
    this->m_data = sh.str();
    return true;
}

/**
 * @brief 将 data 保存到文件中(覆盖写)
 */
void XXSFile::save()
{
    std::fstream fs;
    fs.open(this->m_filename, std::ios::out | std::ios::binary);
    fs << this->m_data;
    fs.close();
}

/**
 * @brief 擦除字符串中指定范围的字符
 * @param[in] start 起始
 * @param[in] len 长度
 */
void XXSFile::erase(int start, int len)
{
    this->m_data.erase(start, len);
}

/**
 * @brief 擦除字符串中某一行的数据
 * @param[in] line 行号, 从 1 开始编号
 */
void XXSFile::erase_line(int line)
{
    XXSStringHelper sh(this->m_data);
    sh.erase_line(line);
    this->m_data = sh.str();
}

/**
 * @brief 在数据的某个位置插入一段字符串
 * @param[in] pos 位置
 * @param[in] str 字符串
 */
void XXSFile::insert(int pos, const std::string &str)
{
    this->m_data.insert(pos, str);
}

/**
 * @brief 在数据末尾追加字符串
 * @param[in] str 字符串
 */
void XXSFile::append(const std::string &str)
{
    this->m_data += str;
}
