//
// Created by Administrator on 2025/7/20.
//

#ifndef XXSLIBRARY_XXSLOGGER_H
#define XXSLIBRARY_XXSLOGGER_H

#include <unordered_map>
#include <chrono>
#include "XXSConsole.h"
#include "XXSFile.h"

/**
 * @brief 日志等级枚举类
 */
enum class XXSLogLevel
{
    Debug,
    Info,
    Warn,
    Error,
    Fatal
};

/**
 * @brief 日志输出位置
 */
enum class XXSLoggerOutputMode
{
    ToConsole = 0b0001,
    ToFile = 0b0010,
    ToConsoleAndFile = 0b0011
};

/**
 * @brief 日志类, 负责输出日志
 */
class XXSLogger
{
private:
    XXSFile m_file;
    XXSLoggerOutputMode m_mode;
    bool m_b_with_timestamp;
public:
    XXSLogger();
    XXSLogger(const std::string &logfile_name, bool b_to_console = false, bool b_with = true);
    ~XXSLogger();

    void set_logfile_name(const std::string &logfile_name);
    void set_logfile_name(const std::string &logfile_name, bool b_to_console);
    void set_output_mode(XXSLoggerOutputMode mode);

    void with_timestamp(bool b_with);

    const std::string &get_logfile_name() const;
    XXSLoggerOutputMode get_output_mode() const;

    void write(XXSLogLevel level, const std::string &info);
    void write_to_console(XXSLogLevel level, const std::string &info);
    void write_to_file(XXSLogLevel level, const std::string &info);
};


#endif //XXSLIBRARY_XXSLOGGER_H
