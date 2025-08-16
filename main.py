import typer
from interactive_numle_solver import InteractiveNumleSolver
import time
import sys
import multiprocessing as mp
from tqdm import tqdm
import numpy as np

# 进程内全局 Solver 缓存，用于复用生成的大规模组合，降低任务初始化开销
_G_SOLVER = None
# 新增：范围批处理工作进程，减少 IPC 与任务调度开销
# 新增：单个谜底的静默求解核心逻辑
def _solve_one_secret(solver: InteractiveNumleSolver, secret_arr: np.ndarray):
    """
    静默模式解算单个谜底，返回是否成功及尝试次数
    """
    # 预先构造一次 ASCII 字符串，避免循环内重复拼接
    secret_str = (secret_arr + 48).tobytes().decode('ascii')

    # 重置解算器状态以复用 all_combinations（不复制）
    solver.possible_combinations = solver.all_combinations
    attempt = 1
    check_fn = InteractiveNumleSolver.check
    L = solver.digit_length
    while True:
        guess = solver.get_next_guess()
        if guess is None:
            # 解算失败
            return False, attempt
        
        result = check_fn(secret_str, guess)
        total_correct = result['total_digits']
        positions_correct = result['correct_positions']
        
        if positions_correct == L:
            # 成功
            return True, attempt
            
        solver.update_possible_combinations(guess, total_correct, positions_correct)
        attempt += 1

# 新增：范围批处理工作进程，减少 IPC 与任务调度开销
def _solve_range_worker(args):
    """
    工作进程函数：处理 [start, end) 范围内的谜底索引，返回聚合统计
    返回: (processed, success_count, total_attempts, max_attempts)
    """
    try:
        start, end, length = args
        global _G_SOLVER
        solver = _G_SOLVER
        if solver is None or getattr(solver, "digit_length", None) != length:
            # 若父进程未预热或长度不匹配，则在子进程按需构建
            solver = InteractiveNumleSolver(length)
            _G_SOLVER = solver

        all_secrets = solver.all_combinations
        
        processed = 0
        success_count = 0
        total_attempts = 0
        max_attempts = 0

        for idx in range(start, end):
            secret_arr = all_secrets[idx]
            is_success, attempts = _solve_one_secret(solver, secret_arr)
            if is_success:
                success_count += 1
                total_attempts += attempts
                if attempts > max_attempts:
                    max_attempts = attempts
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
    求解模式：交互式求解模式。
    """
    solver = InteractiveNumleSolver(length)
    solver.solve_interactive()

@app.command()
def auto_solve(
    secret: str = typer.Argument(None, help="目标数字 (可选，若不提供则随机生成)"),
    length: int = typer.Option(5, "--length", "-l", help="数字长度 (当secret未提供时生效)"),
):
    """
    自动求解模式：给定一个目标数，自动调用求解和检查模式，并且展示过程。
    如果没有输入秘密数字，可以通过-l指定长度（默认5）自动生成一个。
    """
    try:
        if secret is None:
            # 若未提供 secret，则随机生成一个
            print(f"未提供目标数字，将随机生成一个长度为 {length} 的数字。")
            solver = InteractiveNumleSolver(length)
            all_secrets = solver.all_combinations
            secret_idx = np.random.randint(0, len(all_secrets))
            secret_arr = all_secrets[secret_idx]
            secret = (secret_arr + 48).tobytes().decode('ascii')
            print(f"已生成谜底: {secret}")
        else:
            # 使用提供的 secret
            length = len(secret)
            if not secret.isdigit():
                raise ValueError("目标数字必须只包含数字。")
            solver = InteractiveNumleSolver(length)
    except ValueError as e:
        print(f"错误: {e}")
        raise typer.Exit(code=1)

    print(f"进入自动求解模式，目标为: {secret}")
    
    attempt = 1
    start_time = time.time()
    while True:
        guess = solver.get_next_guess()
        if guess is None:
            print("错误：无法找到下一个猜测，可能存在矛盾或已无可能性。")
            break
        
        print(f"\n第 {attempt} 次猜测: {guess}")
        
        result = InteractiveNumleSolver.check(secret, guess)
        total_correct = result['total_digits']
        positions_correct = result['correct_positions']
        print(f"结果: 包含数字: {total_correct}, 位置正确: {positions_correct}")

        if positions_correct == length:
            end_time = time.time()
            print(f"\n成功！在 {attempt} 次猜测后找到答案: {secret}")
            print(f"耗时: {end_time - start_time:.2f} 秒")
            break
        
        solver.update_possible_combinations(guess, total_correct, positions_correct)
        attempt += 1

@app.command()
def play(length: int = typer.Argument(5, help="数字长度")):
    """
    游戏模式：程序随机生成一个数字，由用户来猜。
    """
    try:
        solver = InteractiveNumleSolver(length)
        all_secrets = solver.all_combinations
        secret_idx = np.random.randint(0, len(all_secrets))
        secret_arr = all_secrets[secret_idx]
        secret = (secret_arr + 48).tobytes().decode('ascii')
    except ValueError as e:
        print(f"错误: {e}")
        raise typer.Exit(code=1)

    print(f"游戏开始！我已经想好了一个 {length} 位的数字。")
    print("输入 'q' 放弃并查看答案。")
    
    attempt = 1
    while True:
        guess = typer.prompt(f"第 {attempt} 次猜测").strip()
        if guess.lower() == 'q':
            print(f"游戏结束。答案是: {secret}")
            raise typer.Exit()
        
        try:
            result = InteractiveNumleSolver.check(secret, guess)
            total_correct = result['total_digits']
            positions_correct = result['correct_positions']
            print(f"结果: 包含数字: {total_correct}, 位置正确: {positions_correct}")

            if positions_correct == length:
                print(f"\n恭喜你！在第 {attempt} 次猜测后答对了！答案是: {secret}")
                break
        except ValueError as e:
            print(f"错误: {e}")
        
        attempt += 1

@app.command()
def test(
    length: int = typer.Option(5, "--length", "-l", help="数字长度"),
    processes: int = typer.Option(1, "--processes", "-p", help="进程数；1为单进程，>1启用多进程；0或-1表示使用全部CPU内核"),
    chunksize: int = typer.Option(0, "--chunksize", "-c", help="多进程时的任务分发批大小；<=0 表示自动估算"),
):
    """
    测试模式：自动遍历所有可能性进行测试。支持多进程，并使用 tqdm 显示进度条。
    """
    solver = InteractiveNumleSolver(length)
    print("\n开始自动遍历测试...")
    
    all_secrets = solver.all_combinations
    total_tests = len(all_secrets)
    success_count = 0
    total_attempts = 0
    max_attempts = 0
    processed = 0
    start_time = time.time()

    try:
        if processes is None or processes == 1:
            # 单进程
            for secret in tqdm(all_secrets, desc="测试进度", unit="题"):
                is_success, attempts = _solve_one_secret(solver, secret)
                if is_success:
                    success_count += 1
                    total_attempts += attempts
                    if attempts > max_attempts:
                        max_attempts = attempts
                processed += 1
        else:
            # 多进程：基于索引区间分发任务，避免传输大数组切片；并复用父进程预热的 Solver
            procs = mp.cpu_count() if processes in (0, -1) else max(1, processes)
            procs = min(procs, total_tests)  # 不要创建多于任务数量的进程

            # 预热：将父进程已构造的 Solver 通过 fork 共享到子进程，避免子进程重复构建基表
            global _G_SOLVER
            _G_SOLVER = solver

            # 计算批大小：用户传入优先；否则自动估算为约 procs*4 个批次
            if chunksize and chunksize > 0:
                batch_size = int(chunksize)
            else:
                target_tasks = max(procs * 4, 1)
                batch_size = max(1, (total_tests + target_tasks - 1) // target_tasks)

            # 构造索引区间任务
            tasks = []
            for start_idx in range(0, total_tests, batch_size):
                end_idx = min(start_idx + batch_size, total_tests)
                tasks.append((start_idx, end_idx, length))

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