//
// Created by Administrator on 2025/7/21.
//

#ifndef XXSLIBRARY_XXSFILE_H
#define XXSLIBRARY_XXSFILE_H

#include <fstream>
#include "XXSStringHelper.h"


/**
 * @brief 文件类
 */
class XXSFile
{
private:
    std::string m_filename;
    std::string m_data;
public:
    XXSFile();
    explicit XXSFile(const std::string &filename);
    ~XXSFile();

    void set_filename(const std::string &filename);
    void set_data(const std::string &data);

    const std::string &get_filename() const;
    const std::string &get_data() const;

    /**
     * @brief 加载文件数据进 data
     * @return 是否成功加载文件
     */
    bool load();
    void save();

    // 以下四个函数操作的是 data! 不是文件本身! 是把文件 load 到 data 后操作 data, 再把它 write 进文件

    void erase(int start, int len);
    void erase_line(int line);
    void insert(int pos, const std::string &str);
    void append(const std::string &str);
};


#endif //XXSLIBRARY_XXSFILE_H
