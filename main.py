from interactive_numle_solver import InteractiveNumleSolver
import time
import sys

if __name__ == "__main__":
    print("Numle 助手 - 选择模式:")
    print("1. 检查模式 (check)")
    print("2. 求解模式 (solve)")
    print("3. 自动遍历测试")
    mode = input("请输入模式编号: ").strip()
    
    if mode == "1":
        secret = input("请输入谜底数字: ").strip()
        print("\n进入检查模式，输入'q'退出")
        while True:
            guess = input("请输入猜测数字: ").strip()
            if guess.lower() == 'q':
                print("退出检查模式")
                break
                
            try:
                result = InteractiveNumleSolver.check(secret, guess)
                print(f"结果: 包含数字: {result['total_digits']}, 位置正确: {result['correct_positions']}")
            except ValueError as e:
                print(f"错误: {e}")
    elif mode == "2":
        length = int(input("请输入数字长度: "))
        solver = InteractiveNumleSolver(length)
        solver.solve_interactive()
    elif mode == "3":
        length = int(input("请输入数字长度: "))
        solver = InteractiveNumleSolver(length)
        print("\n开始自动遍历测试...")
        
        start_time = time.time()
        total_tests = len(solver.all_combinations)
        success_count = 0
        total_attempts = 0
        max_attempts = 0
        
        current_test = 0
        try:
            for secret in solver.all_combinations:
                current_test += 1
                print(f"进度: {current_test}/{total_tests}", end="\r")
                solver.possible_combinations = solver.all_combinations.copy()
                attempt = 1
                solved = False
                
                while True:
                    guess = solver.get_next_guess()
                    if guess is None:
                        break
                        
                    result = solver.check(''.join(map(str, secret)), guess)
                    total_correct = result['total_digits']
                    positions_correct = result['correct_positions']
                    
                    if positions_correct == solver.digit_length:
                        solved = True
                        success_count += 1
                        total_attempts += attempt
                        if attempt > max_attempts:
                            max_attempts = attempt
                        break
                        
                    solver.update_possible_combinations(guess, total_correct, positions_correct)
                    attempt += 1
        except KeyboardInterrupt:
            print("\n\n测试被中断，显示当前统计结果:")
            end_time = time.time()
            total_time = end_time - start_time
            print(f"已完成测试数: {current_test}/{total_tests}")
            print(f"成功次数: {success_count}")
            if current_test > 0:
                print(f"当前成功率: {success_count/current_test*100:.2f}%")
            if success_count > 0:
                print(f"平均猜测次数: {total_attempts/success_count:.2f}")
                print(f"最高猜测次数: {max_attempts}")
            print(f"当前总耗时: {total_time:.2f}秒")
            if current_test > 0:
                print(f"平均每个测试耗时: {total_time/current_test:.4f}秒")
            sys.exit(0)
                
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n测试完成！结果:")
        print(f"测试总数: {total_tests}")
        print(f"成功次数: {success_count}")
        print(f"成功率: {success_count/total_tests*100:.2f}%")
        if success_count > 0:
            print(f"平均猜测次数: {total_attempts/success_count:.2f}")
            print(f"最高猜测次数: {max_attempts}")
        print(f"总耗时: {total_time:.2f}秒")
        if total_tests > 0:
            print(f"平均每个测试耗时: {total_time/total_tests:.4f}秒")
    else:
        print("无效模式选择")