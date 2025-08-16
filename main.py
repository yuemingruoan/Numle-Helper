import typer
from interactive_numle_solver import (
    InteractiveNumleSolver, _check_nb, _calculate_entropies_nb, _filter_combinations_nb
)
import time
import threading
import numpy as np
import numba
from tqdm import tqdm

# ======================================================================================
# Numba JIT 优化的核心求解器
# ======================================================================================

@numba.njit(cache=True)
def _solve_one_secret_nb(all_combinations, secret_arr):
    """
    Numba JIT: 静默模式解算单个谜底，返回是否成功及尝试次数
    - 直接操作 numpy 数组，无 Python 对象开销
    """
    L = all_combinations.shape[1]
    possible_combinations = all_combinations.copy() # 每个求解过程有自己的独立副本
    
    attempt = 1
    while True:
        # 1. 获取猜测 (内联 get_next_guess 逻辑)
        if len(possible_combinations) == 0:
            return False, attempt # 解算失败

        entropies = _calculate_entropies_nb(possible_combinations, L)
        best_idx = np.argmax(entropies)
        guess_arr = possible_combinations[best_idx]

        # 2. 检查 (内联 check 逻辑)
        total_correct, positions_correct = _check_nb(secret_arr, guess_arr)
        
        if positions_correct == L:
            return True, attempt # 成功
            
        # 3. 更新可能性 (内联 update_possible_combinations 逻辑)
        mask = _filter_combinations_nb(
            possible_combinations, guess_arr, total_correct, positions_correct
        )
        possible_combinations = possible_combinations[mask]
        
        attempt += 1

@numba.njit(parallel=True, cache=True)
def _test_all_nb(all_combinations, results):
    """
    Numba JIT (并行模式): 测试所有组合
    直接在传入的 results 数组上修改，方便外部跟踪进度
    """
    n_tests = len(all_combinations)
    for i in numba.prange(n_tests):
        secret = all_combinations[i]
        is_success, attempts = _solve_one_secret_nb(all_combinations, secret)
        results[i, 0] = is_success
        results[i, 1] = attempts

# ======================================================================================
# Python 侧的包装与工作流
# ======================================================================================

def _solve_one_secret(solver: InteractiveNumleSolver, secret_arr: np.ndarray):
    """
    静默求解的 Python 包装器，调用 Numba JIT 核心
    """
    # 直接调用 JIT 函数
    return _solve_one_secret_nb(solver.all_combinations, secret_arr)

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

    # Numba JIT 预热
    print("Numba JIT 预热中...")
    start_time = time.time()
    # 预热时传入一个空的 results 数组
    dummy_results = np.empty((1, 2), dtype=np.int32)
    if parallel:
        _test_all_nb(all_secrets[:1], dummy_results)
    else:
        _solve_one_secret_nb(all_secrets, all_secrets[0])
    end_time = time.time()
    print(f"预热完成，耗时: {end_time - start_time:.2f} 秒。")

    # 执行主计算
    start_time = time.time()
    
    if parallel:
        # Numba 并行模式 + 原生进度条
        print("正在执行并行计算...")
        # 初始化 results 数组，使用 -1 作为未完成任务的标记
        results = np.full((total_tests, 2), -1, dtype=np.int32)
        
        # 在工作线程中运行 Numba 计算
        worker = threading.Thread(target=_test_all_nb, args=(all_secrets, results))
        worker.start()
        
        # 主线程负责更新进度
        while worker.is_alive():
            # 通过检查标记值来计算已完成的任务数
            processed = np.sum(results[:, 1] != -1)
            print(f"\r进度: {processed}/{total_tests}", end="")
            time.sleep(0.2)
        
        worker.join()
        # 确保最终进度显示为 100%
        print(f"\r进度: {total_tests}/{total_tests}")

    else:
        # 单线程 tqdm 模式
        results = np.empty((total_tests, 2), dtype=np.int32)
        for i in tqdm(range(total_tests), desc="测试进度", unit="题"):
            is_success, attempts = _solve_one_secret_nb(all_secrets, all_secrets[i])
            results[i, 0] = is_success
            results[i, 1] = attempts

    end_time = time.time()
    total_time = end_time - start_time
    print("计算完成。")

    # 结果统计 (使用 Numpy 高效完成)
    success_mask = results[:, 0] == 1
    success_count = np.sum(success_mask)
    
    print(f"\n测试完成！结果:")
    print(f"测试总数: {total_tests}")
    print(f"成功次数: {success_count}")
    
    if total_tests > 0:
        print(f"成功率: {success_count / total_tests * 100:.2f}%")
        
    if success_count > 0:
        successful_attempts = results[success_mask, 1]
        total_attempts = np.sum(successful_attempts)
        max_attempts = np.max(successful_attempts)
        print(f"平均猜测次数: {total_attempts / success_count:.2f}")
        print(f"最高猜测次数: {max_attempts}")
        
    print(f"总耗时: {total_time:.2f}秒")
    if total_tests > 0:
        print(f"平均每个测试耗时: {total_time / total_tests:.4f}秒")

if __name__ == "__main__":
    app()