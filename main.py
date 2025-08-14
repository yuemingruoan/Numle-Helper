import typer
from interactive_numle_solver import InteractiveNumleSolver
import time
import sys
import multiprocessing as mp
from tqdm import tqdm

def _solve_secret_worker(args):
    """
    工作进程函数：对单个谜底执行求解，返回 (success, attempts)
    """
    secret, length = args
    solver = InteractiveNumleSolver(length)
    solver.possible_combinations = solver.all_combinations.copy()
    attempt = 1
    while True:
        guess = solver.get_next_guess()
        if guess is None:
            return False, attempt - 1
        result = InteractiveNumleSolver.check(''.join(map(str, secret)), guess)
        total_correct = result['total_digits']
        positions_correct = result['correct_positions']
        if positions_correct == solver.digit_length:
            return True, attempt
        solver.update_possible_combinations(guess, total_correct, positions_correct)
        attempt += 1

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
def test(
    length: int = typer.Option(5, "--length", "-l", help="数字长度"),
    processes: int = typer.Option(1, "--processes", "-p", help="进程数；1为单进程，>1启用多进程；0或-1表示使用全部CPU内核"),
    chunksize: int = typer.Option(50, "--chunksize", "-c", help="多进程时的任务分发批大小"),
):
    """
    自动遍历所有可能性进行测试。支持多进程，并使用 tqdm 显示进度条。
    """
    solver = InteractiveNumleSolver(length)
    print("\n开始自动遍历测试...")
    
    start_time = time.time()
    all_secrets = solver.all_combinations
    total_tests = len(all_secrets)
    success_count = 0
    total_attempts = 0
    max_attempts = 0
    processed = 0

    try:
        if processes is None or processes == 1:
            # 单进程
            for secret in tqdm(all_secrets, desc="测试进度", unit="题"):
                # 重置解算器状态
                solver.possible_combinations = solver.all_combinations.copy()
                attempt = 1
                while True:
                    guess = solver.get_next_guess()
                    if guess is None:
                        break
                    result = InteractiveNumleSolver.check(''.join(map(str, secret)), guess)
                    total_correct = result['total_digits']
                    positions_correct = result['correct_positions']
                    
                    if positions_correct == solver.digit_length:
                        success_count += 1
                        total_attempts += attempt
                        if attempt > max_attempts:
                            max_attempts = attempt
                        break
                        
                    solver.update_possible_combinations(guess, total_correct, positions_correct)
                    attempt += 1
                processed += 1
        else:
            # 多进程
            procs = mp.cpu_count() if processes in (0, -1) else max(1, processes)
            with mp.get_context("fork").Pool(processes=procs) as pool:
                with tqdm(total=total_tests, desc="测试进度", unit="题") as pbar:
                    for success, attempts in pool.imap_unordered(
                        _solve_secret_worker,
                        ((secret, length) for secret in all_secrets),
                        chunksize=chunksize,
                    ):
                        processed += 1
                        pbar.update(1)
                        if success:
                            success_count += 1
                            total_attempts += attempts
                            if attempts > max_attempts:
                                max_attempts = attempts
    except KeyboardInterrupt:
        end_time = time.time()
        total_time = end_time - start_time
        print("\n\n测试被中断，显示当前统计结果:")
        print(f"已完成测试数: {processed}/{total_tests}")
        print(f"成功次数: {success_count}")
        if processed > 0:
            print(f"当前成功率: {success_count/processed*100:.2f}%")
        if success_count > 0:
            print(f"平均猜测次数: {total_attempts/success_count:.2f}")
            print(f"最高猜测次数: {max_attempts}")
        print(f"当前总耗时: {total_time:.2f}秒")
        if processed > 0:
            print(f"平均每个测试耗时: {total_time/processed:.4f}秒")
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