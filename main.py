# ======================================================================================
# Numba JIT 优化的核心求解器
# ======================================================================================

import typer
from interactive_numle_solver import (
    InteractiveNumleSolver, _check_nb, _calculate_entropies_nb,
    _filter_combinations_nb_inplace, _get_best_guess_from_mask_nb
)
import time
import threading
import numpy as np
import numba
from tqdm import tqdm

# ======================================================================================
# Numba JIT 优化的核心求解器
# ======================================================================================

@numba.njit(cache=True, fastmath=True)
def _solve_one_secret_nb(all_combinations, secret_arr, first_guess_arr=None):
    """
    Numba JIT: 静默模式解算单个谜底，返回是否成功及尝试次数
    - 使用 mask 避免数组复制，性能更高
    - 接受可选的 first_guess_arr 以免重复计算
    """
    L = all_combinations.shape[1]
    n_combinations = all_combinations.shape[0]
    
    # 使用掩码代替数组复制
    mask = np.ones(n_combinations, dtype=np.bool_)
    
    # 初始化 guess_arr 以消除 Pylance 警告
    guess_arr = all_combinations[0]
    
    attempt = 1
    while True:
        # 1. 获取猜测
        if np.sum(mask) == 0:
            return False, attempt  # 解算失败

        # 如果是第一次尝试且提供了预计算的猜测，直接使用
        if attempt == 1 and first_guess_arr is not None:
            guess_arr = first_guess_arr
        else:
            # 高效地获取最佳猜测
            best_idx = _get_best_guess_from_mask_nb(all_combinations, mask, L)
            if best_idx == -1:
                return False, attempt # 无法找到猜测
            guess_arr = all_combinations[best_idx]

        # 2. 检查
        total_correct, positions_correct = _check_nb(secret_arr, guess_arr)
        
        if positions_correct == L:
            return True, attempt # 成功
            
        # 3. 原地更新掩码
        _filter_combinations_nb_inplace(
            mask, all_combinations, guess_arr, total_correct, positions_correct
        )
        
        attempt += 1

@numba.njit(parallel=True, cache=True, fastmath=True)
def _test_all_nb(all_combinations, results, first_guess_arr):
    """
    Numba JIT (并行模式): 测试所有组合
    - 接收预先计算好的 first_guess_arr
    """
    n_tests = len(all_combinations)
    for i in numba.prange(n_tests):
        secret = all_combinations[i]
        is_success, attempts = _solve_one_secret_nb(all_combinations, secret, first_guess_arr)
        results[i, 0] = is_success
        results[i, 1] = attempts

@numba.njit(cache=True, fastmath=True)
def _test_all_nb_single_thread(all_combinations, results, first_guess_arr):
    """
    Numba JIT (单线程): 测试所有组合，以消除 Python 循环开销
    """
    n_tests = len(all_combinations)
    for i in range(n_tests):
        secret = all_combinations[i]
        is_success, attempts = _solve_one_secret_nb(all_combinations, secret, first_guess_arr)
        results[i, 0] = is_success
        results[i, 1] = attempts

# ======================================================================================
# Python 侧的包装与工作流
# ======================================================================================


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
        
        if len(set(guess)) != len(guess):
            print("错误：猜测的数字不能包含重复值。")
            continue
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
            if not secret.isdigit() or len(set(secret)) != length:
                raise ValueError("目标数字必须只包含数字，且不能有重复。")
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
            if len(set(guess)) != len(guess):
                print("错误：猜测的数字不能包含重复值。")
                continue

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
    parallel: bool = typer.Option(False, "--parallel/--no-parallel", "-p", help="是否启用 Numba 并行计算"),
):
    """
    测试模式：使用 Numba 并行计算自动遍历所有可能性。
    """
    solver = InteractiveNumleSolver(length)
    all_secrets = solver.all_combinations
    total_tests = len(all_secrets)

    print("\n开始自动遍历测试...")
    print(f"数字长度: {length}, 总测试数: {total_tests}, 并行计算: {'启用' if parallel else '禁用'}")
    if parallel:
        print(f"Numba 并行数量: {numba.get_num_threads()}")

    # Numba JIT 预热
    print("Numba JIT 预热中...")
    start_time = time.time()
    
    # 预热时需要计算首猜
    entropies = _calculate_entropies_nb(all_secrets, length)
    first_guess_idx = np.argmax(entropies)
    first_guess_arr = all_secrets[first_guess_idx]
    
    # 预热 JIT 函数
    dummy_results = np.empty((1, 2), dtype=np.int32)
    _test_all_nb(all_secrets[:1], dummy_results, first_guess_arr) # 并行预热
    _solve_one_secret_nb(all_secrets, all_secrets[0], first_guess_arr) # 单核求解器预热
    _test_all_nb_single_thread(all_secrets[:1], dummy_results, first_guess_arr) # 单线程测试循环预热

    end_time = time.time()
    print(f"预热完成，耗时: {end_time - start_time:.2f} 秒。")
    print(f"预计算出的最佳首次猜测: {''.join(map(str, first_guess_arr))}")

    # --- 执行主计算 ---
    start_time = time.time()
    results = np.empty((total_tests, 2), dtype=np.int32)
    results.fill(-1)
    
    target_func = _test_all_nb if parallel else _test_all_nb_single_thread
    desc = "并行测试进度" if parallel else "单线程测试进度"

    worker = threading.Thread(target=target_func, args=(all_secrets, results, first_guess_arr))
    worker.daemon = True  # 设置为守护线程，以便主线程退出时可以强制终止
    worker.start()
    
    interrupted = False
    try:
        with tqdm(total=total_tests, desc=desc) as pbar:
            while worker.is_alive():
                processed = np.sum(results[:, 1] != -1)
                pbar.update(processed - pbar.n)
                time.sleep(0.1)  # 短暂休眠以降低 CPU 占用
            # 确保在计算结束后，进度条能更新到100%
            processed = np.sum(results[:, 1] != -1)
            pbar.update(processed - pbar.n)
    except KeyboardInterrupt:
        interrupted = True
        print("\n\n测试被用户中断。正在处理已完成部分的结果...")
    
    end_time = time.time()
    total_time = end_time - start_time
    if not interrupted:
        print("计算完成。")

    # --- 结果统计 ---
    processed_mask = results[:, 1] != -1
    total_processed = int(np.sum(processed_mask))

    if total_processed == 0:
        print("警告：没有完成任何测试。")
        raise typer.Exit(code=1)

    # 在已处理的测试中筛选成功和失败
    success_mask = (results[:, 0] == 1) & processed_mask
    fail_mask = (results[:, 0] == 0) & processed_mask

    success_count = np.sum(success_mask)
    fail_count = np.sum(fail_mask)

    print(f"\n--- 测试结果 ---")
    print(f"原定测试总数: {total_tests}")
    print(f"已完成测试数: {total_processed} ({total_processed / total_tests * 100:.2f}%)")
    
    if fail_count > 0:
        print(f"失败次数: {fail_count}")
        success_rate = success_count / total_processed * 100
        print(f"成功率 (基于已完成部分): {success_rate:.2f}%")

    if success_count > 0:
        successful_attempts = results[success_mask, 1]
        avg_attempts = np.mean(successful_attempts)
        max_attempts = np.max(successful_attempts)
        
        print(f"平均猜测次数: {avg_attempts:.2f}")
        print(f"最高猜测次数: {max_attempts}")

        # 猜测次数分布
        print("\n猜测次数分布:")
        # 使用bincount获得每个尝试次数的频率
        attempts_distribution = np.bincount(successful_attempts)
        for i, count in enumerate(attempts_distribution):
            if i > 0 and count > 0: # 从1次开始，且次数大于0
                percentage = count / success_count * 100
                print(f"  {i} 次: {count:5d} 个 ({percentage:5.2f}%)")
        
        # 找到需要最多次猜测的谜底
        hardest_secrets_mask = (results[:, 1] == max_attempts) & success_mask
        hardest_secrets_indices = np.where(hardest_secrets_mask)[0]
        
        display_limit = 5
        print(f"\n需要 {max_attempts} 次猜测的谜底 (最多显示 {min(display_limit, len(hardest_secrets_indices))} 个):")
        for i, secret_idx in enumerate(hardest_secrets_indices):
            if i >= display_limit:
                print(f"  ... (及其他 {len(hardest_secrets_indices) - display_limit} 个)")
                break
            secret_arr = all_secrets[secret_idx]
            secret_str = ''.join(map(str, secret_arr))
            print(f"  - {secret_str}")

    if fail_count > 0:
        failed_secrets_indices = np.where(fail_mask)[0]
        display_limit = 5
        print(f"\n失败的谜底 (最多显示 {min(display_limit, len(failed_secrets_indices))} 个):")
        for i, secret_idx in enumerate(failed_secrets_indices):
            if i >= display_limit:
                print(f"  ... (及其他 {len(failed_secrets_indices) - display_limit} 个)")
                break
            secret_arr = all_secrets[secret_idx]
            secret_str = ''.join(map(str, secret_arr))
            failed_attempts = results[secret_idx, 1]
            print(f"  - {secret_str} (在 {failed_attempts} 次尝试后失败)")

    print(f"\n总耗时: {total_time:.2f}秒")
    if total_processed > 0:
        total_guesses = np.sum(results[processed_mask, 1])
        problems_per_second = total_processed / total_time
        guesses_per_second = total_guesses / total_time
        
        print(f"平均每个测试耗时: {total_time / total_processed:.4f}秒")
        print(f"每秒解题数 (TPS): {problems_per_second:.2f}")
        print(f"每秒猜测数 (GPS): {guesses_per_second:.2f}")

if __name__ == "__main__":
    app()