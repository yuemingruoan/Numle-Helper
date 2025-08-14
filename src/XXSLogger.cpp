//
// Created by Administrator on 2025/7/20.
//

#include "XXSLogger.h"

/**
 * @brief 日志等级到对应的标识的映射
 */
std::unordered_map<XXSLogLevel, std::string> g_loglevel_map =
    {
        {XXSLogLevel::Debug,    "[DEBUG] "},
        {XXSLogLevel::Info,     "[INFO]  "},
        {XXSLogLevel::Warn,     "[WARN]  "},
        {XXSLogLevel::Error,    "[ERROR] "},
        {XXSLogLevel::Fatal,    "[FATAL] "}
    };

/**
 * @brief 日志等级到对应输出颜色的映射
 */
std::unordered_map<XXSLogLevel, long long> g_color_map =
    {
        {XXSLogLevel::Debug,    0x808080},
        {XXSLogLevel::Info,     0x008000},
        {XXSLogLevel::Warn,     0xffa500},
        {XXSLogLevel::Error,    0xff0000},
        {XXSLogLevel::Fatal,    0x8b0000}
    };

XXSLogger::XXSLogger()
{
    this->m_file.set_filename("Log.txt");
    this->m_mode = XXSLoggerOutputMode::ToConsole;
    this->m_b_with_timestamp = true;
}

/**
 * @brief 用日志文件名来初始化
 * @param[in] logfile_name 日志文件名
 * @param[in] b_to_console 是否输出到控制台, 默认为否
 * @param[in] b_with 是否带着时间戳, 默认为是
 */
XXSLogger::XXSLogger(const std::string &logfile_name, bool b_to_console, bool b_with)
{
    this->m_file.set_filename(logfile_name);
    if(b_to_console)
    {
        this->m_mode = XXSLoggerOutputMode::ToConsoleAndFile;
    }
    else
    {
        this->m_mode = XXSLoggerOutputMode::ToFile;
    }
    this->m_b_with_timestamp = b_with;
}

XXSLogger::~XXSLogger() = default;

/**
 * @brief 设置日志文件名
 * @param[in] logfile_name 日志文件名
 * @param[in] b_to_console 是否输出到控制台
 */
void XXSLogger::set_logfile_name(const std::string &logfile_name, bool b_to_console)
{
    this->m_file.set_filename(logfile_name);
    if(b_to_console)
    {
        this->m_mode = XXSLoggerOutputMode::ToConsoleAndFile;
    }
    else
    {
        this->m_mode = XXSLoggerOutputMode::ToFile;
    }
}

/**
 * @brief 设置日志输出模式
 * @param[in] mode 模式
 */
void XXSLogger::set_output_mode(XXSLoggerOutputMode mode)
{
    this->m_mode = mode;
}

/**
 * @brief 设置日志文件名
 * @param[in] logfile_name 文件名
 */
void XXSLogger::set_logfile_name(const std::string &logfile_name)
{
    this->m_file.set_filename(logfile_name);
}

/**
 * @brief 获得日志文件名
 * @return 日志文件名
 */
const std::string &XXSLogger::get_logfile_name() const
{
    return this->m_file.get_filename();
}

/**
 * @brief 获得日志文件输出模式
 * @return 输出模式
 */
XXSLoggerOutputMode XXSLogger::get_output_mode() const
{
    return this->m_mode;
}

/**
 * @brief 输出一条日志
 * @param[in] level 日志等级
 * @param[in] info 日志信息
 */
void XXSLogger::write(XXSLogLevel level, const std::string &info)
{
    bool b_console = static_cast<long>(this->m_mode) & 0b0001;
    bool b_file = static_cast<long>(this->m_mode) & 0b0010;
    if(b_console)
    {
        this->write_to_console(level, info);
    }
    if(b_file)
    {
        this->write_to_file(level, info);
    }
}

/**
 * @brief 向控制台输出一条日志
 * @param[in] level 日志等级
 * @param[in] info 日志信息
 */
void XXSLogger::write_to_console(XXSLogLevel level, const std::string &info)
{
    XXSConsole con;
    if(this->m_b_with_timestamp)
    {
        auto chr_now = std::chrono::system_clock::now();
        std::time_t time_now = std::chrono::system_clock::to_time_t(chr_now);
        con.write("(").write(std::put_time(std::localtime(&time_now), "%Y-%m-%d %H:%M:%S")).write(")");
    }
    con.set_text_color(g_color_map[level]).write(g_loglevel_map[level]).reset_color();
    con.write_line(info);
}

/**
 * @brief 向文件输出一条日志
 * @param[in] level 日志等级
 * @param[in] info 日志信息
 */
void XXSLogger::write_to_file(XXSLogLevel level, const std::string &info)
{
    this->m_file.load();
    if(this->m_b_with_timestamp)
    {
        auto chr_now = std::chrono::system_clock::now();
        std::time_t time_now = std::chrono::system_clock::to_time_t(chr_now);
        XXSStringHelper sh;
        sh.write("(").write(std::put_time(std::localtime(&time_now), "%Y-%m-%d %H:%M:%S")).write(")");
        this->m_file.append(sh.str());
    }
    this->m_file.append(g_loglevel_map[level]);
    this->m_file.append(info);
    this->m_file.save();
}

/**
 * @brief 设置日志输出是否带时间戳
 * @param[in] b_with 是否带时间戳
 */
void XXSLogger::with_timestamp(bool b_with)
{
    this->m_b_with_timestamp = b_with;
}
