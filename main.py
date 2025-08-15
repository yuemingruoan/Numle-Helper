import typer
from interactive_numle_solver import InteractiveNumleSolver
import time
import sys
import multiprocessing as mp
from tqdm import tqdm

# 进程内全局 Solver 缓存，用于复用生成的大规模组合，降低任务初始化开销
_G_SOLVER = None
# 新增：范围批处理工作进程，减少 IPC 与任务调度开销
def _solve_range_worker(args):
    """
    工作进程函数：一次性处理一段谜底范围，返回聚合统计
    返回: (processed, success_count, total_attempts, max_attempts)
    """
    try:
        secrets_slice, length = args
        global _G_SOLVER
        solver = _G_SOLVER
        if solver is None or getattr(solver, "digit_length", None) != length:
            solver = InteractiveNumleSolver(length)
            _G_SOLVER = solver

        processed = 0
        success_count = 0
        total_attempts = 0
        max_attempts = 0

        for secret in secrets_slice:
            # 重置解算器状态以复用 all_combinations
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

        return processed, success_count, total_attempts, max_attempts
    except KeyboardInterrupt:
        return 0, 0, 0, 0

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
    chunksize: int = typer.Option(0, "--chunksize", "-c", help="多进程时的任务分发批大小；<=0 表示自动估算"),
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
            # 多进程：将全部可能谜底按进程数等分，分别一次性下发到各进程处理
            procs = mp.cpu_count() if processes in (0, -1) else max(1, processes)
            procs = min(procs, total_tests)  # 不要创建多于任务数量的进程
            # 预先切片（一次性传完其被分配的范围）
            size = (total_tests + procs - 1) // procs  # 向上取整
            tasks = [ (all_secrets[i:i+size], length) for i in range(0, total_tests, size) ]
            with mp.get_context("fork").Pool(processes=procs) as pool:
                with tqdm(total=total_tests, desc="测试进度", unit="题") as pbar:
                    for processed_i, success_i, attempts_i, max_attempts_i in pool.imap_unordered(
                        _solve_range_worker,
                        tasks,
                        chunksize=1,
                    ):
                        processed += processed_i
                        pbar.update(processed_i)
                        success_count += success_i
                        total_attempts += attempts_i
                        if max_attempts_i > max_attempts:
                            max_attempts = max_attempts_i
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