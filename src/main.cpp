#include <iostream>
#include "XXSStringHelper.h"
#include "XXSConsole.h"
#include "XXSFile.h"
#include "XXSLogger.h"
#include "XXSNumleChecker.h"
#include "XXSNumleSolver.h"
#include "gpu/GpuSolverFactory.h"
#include <cctype>

std::vector<std::string> get_line_parts(XXSConsole &con)
{
    con.write("> ");

    std::string input_line;
    con.get_line(input_line);
    XXSStringHelper sh(input_line);
    std::string part;
    std::vector<std::string> v_parts;

    while(sh.read(part))
    {
        v_parts.push_back(part);
    }
    return v_parts;
}

int main()
{
    XXSConsole::init_encoding();
    XXSConsole::enable_ansi_esc_codes(true);
    XXSConsole con;

    // 在程序启动时显示计算后端信息
    {
        auto solver = ::GpuSolverFactory::createSolver();
        con.write("使用计算后端: ").write(solver->getBackendName()).new_line();
    }

    while(true)
    {
        con.reset_color();

        auto v_parts = get_line_parts(con);

        if(v_parts.empty())
        {
            continue;
        }

        if(v_parts[0] == "quit")
        {
            con.write_line("Bye!");
            break;
        }

        if(v_parts[0] == "check")
        {
            con.set_text_color(0xffff00);

            std::string num;
            con.write("Check mode : ");
            if(v_parts.size() == 1)
            {
                con.get_line(num);
                XXSStringHelper num_getter(num);
                long long inum;
                if(num_getter.read(inum))
                {
                    num = std::to_string(inum);
                }
            }
            else
            {
                num = v_parts[1];
                con.write_line(num);
            }

            XXSNumleChecker nc(num);

            while(true)
            {
                auto subcmd = get_line_parts(con);

                if(subcmd.empty())
                {
                    continue;
                }

                if(subcmd[0] == "back")
                {
                    con.write_line("Switched to main mode.");
                    break;
                }

                con.write("Contained : ").write(nc.contained(subcmd[0]));
                con.write("  Matching : ").write(nc.matching(subcmd[0])).new_line();
            }
        }
        if(v_parts[0] == "solve")
        {
            con.set_text_color(0x00d9ff);

            int n = 5;
            if(v_parts.size() > 1)
            {
                n = std::stoi(v_parts[1]);
            }
            con.write("Solver mode : ").write(n).write_line(" digits");

            XXSNumleSolver ns(n);
            ns.init_set();

            while(true)
            {
                auto subcmd = get_line_parts(con);

                if(subcmd.empty())
                {
                    continue;
                }

                if(subcmd[0] == "back")
                {
                    con.write_line("Switched to main mode.");
                    break;
                }
                if(subcmd[0] == "list")
                {
                    auto set = ns.get_set();

                    for(auto &v : set)
                    {
                        con.write(v).write(" ");
                    }
                    con.new_line();

                    continue;
                }
                if(subcmd[0] == "size")
                {
                    con.write(ns.get_set_size()).new_line();
                    continue;
                }
                if(subcmd[0] == "entropy")
                {
                    std::string guess;
                    if(subcmd.size() == 1)
                    {
                        con.write("Please input a guess : ");
                        con.get_line(guess);
                        XXSStringHelper num_getter(guess);
                        long long inum;
                        if(num_getter.read(inum))
                        {
                            guess = std::to_string(inum);
                        }
                    }
                    else
                    {
                        guess = subcmd[1];
                    }
                    long double entropy = ns.get_guess_entropy(guess);
                    con.write("Entropy : ").write(entropy).new_line();
                    continue;
                }
                if(subcmd[0] == "best")
                {
                    auto best_guess_entropy = ns.get_best_guess(true);
                    con.write("Best guess : ").write(best_guess_entropy.first);
                    con.write("   Entropy : ").write(best_guess_entropy.second).new_line();
                    continue;
                }
                if(subcmd[0] == "read_table")
                {
                    std::string filename = "table.txt";
                    if(subcmd.size() > 1)
                    {
                        filename = subcmd[1];
                    }
                    con.write("Read table file : ").write_line(filename);
                    ns.read_table(filename);
                    auto table = ns.get_table();
                    for(auto &p : table)
                    {
                        auto history = p.first;
                        auto guesses = p.second;
                        con.write("History : ");
                        for(auto &h : history)
                        {
                            con.write("[").write(h.first).write(" ");
                            con.write(h.second.first).write(" ").write(h.second.second);
                            con.write("] ");
                        }
                        con.write("  Numbers : ");
                        for(auto &ge : guesses)
                        {
                            con.write("[").write(ge.first).write(" ").write(ge.second).write("] ");
                        }
                        con.new_line();
                    }
                    continue;
                }
                if(subcmd.size() >= 3)
                {
                    std::string guess = subcmd[0];
                    int contained = std::stoi(subcmd[1]);
                    int matching = std::stoi(subcmd[2]);

                    // 验证提示是否与当前解集相容
                    if (!XXSNumleSolver::is_valid_hint(ns.get_set(), guess, contained, matching)) {
                        con.set_text_color(0xffff00); // 黄色警告
                        con.write("[WARN] 输入的提示与当前可能解集不相容!").new_line();
                        con.write("      这可能导致解集变空，请检查输入是否正确").new_line();
                        con.reset_color();
                    }
                    
                    long double info_content = ns.restrict(guess, contained, matching);
                    con.write("Remained possibles : ").write(ns.get_set_size()).new_line();
                    con.write("Information Content : ").write(info_content).new_line();
                }
            }
        }
    }


    return 0;
}
