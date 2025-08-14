import typer
from interactive_numle_solver import InteractiveNumleSolver
import time
import sys

app = typer.Typer()

@app.command()
def check(secret: str = typer.Argument(..., help="谜底数字")):
    """
    检查模式：根据给定的谜底，检查猜测的数字。
    """
    print(f"进入检查模式，谜底为: {secret}")
    print("输入 'q' 退出")
    while True:
        guess = typer.prompt("请输入猜测数字").strip()
        if guess.lower() == 'q':
            print("退出检查模式")
            raise typer.Exit()
        
        try:
            result = InteractiveNumleSolver.check(secret, guess)
            print(f"结果: 包含数字: {result['total_digits']}, 位置正确: {result['correct_positions']}")
        except ValueError as e:
            print(f"错误: {e}")

@app.command()
def solve(length: int = typer.Option(5, "--length", "-l", help="数字长度")):
    """
    交互式求解模式。
    """
    solver = InteractiveNumleSolver(length)
    solver.solve_interactive()

@app.command()
def test(length: int = typer.Option(5, "--length", "-l", help="数字长度")):
    """
    自动遍历所有可能性进行测试。
    """
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
            # 使用 f-string 和 \r 实现动态更新
            print(f"进度: {current_test}/{total_tests}", end="\r", flush=True)
            
            # 重置解算器状态
            solver.possible_combinations = solver.all_combinations.copy()
            attempt = 1
            solved = False
            
            while True:
                guess = solver.get_next_guess()
                if guess is None:
                    break
                    
                result = InteractiveNumleSolver.check(''.join(map(str, secret)), guess)
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
    if total_tests > 0:
        print(f"成功率: {success_count/total_tests*100:.2f}%")
    if success_count > 0:
        print(f"平均猜测次数: {total_attempts/success_count:.2f}")
        print(f"最高猜测次数: {max_attempts}")
    print(f"总耗时: {total_time:.2f}秒")
    if total_tests > 0:
        print(f"平均每个测试耗时: {total_time/total_tests:.4f}秒")

if __name__ == "__main__":
    app()