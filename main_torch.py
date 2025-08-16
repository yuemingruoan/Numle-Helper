import typer
from torch_numle_solver import TorchNumleSolver
import time
import torch
from tqdm import tqdm

# ======================================================================================
# PyTorch 优化的核心求解器
# ======================================================================================

def _solve_one_secret_torch(solver: TorchNumleSolver, secret_arr: torch.Tensor):
    """
    静默模式解算单个谜底 (PyTorch 版本)
    - 直接操作 torch.Tensor
    """
    # 每个求解过程有自己的独立副本
    possible_combinations = solver.all_combinations.clone()
    
    attempt = 1
    with torch.no_grad():
        while True:
            # 1. 获取猜测
            if len(possible_combinations) == 0:
                return False, attempt # 解算失败

            entropies = TorchNumleSolver._calculate_entropies_torch(possible_combinations)
            best_idx = torch.argmax(entropies)
            guess_arr = possible_combinations[best_idx]

            # 2. 检查
            total_correct, positions_correct = TorchNumleSolver._check_torch(secret_arr, guess_arr)
            
            if positions_correct == solver.digit_length:
                return True, attempt # 成功
                
            # 3. 更新可能性
            mask = TorchNumleSolver._filter_combinations_torch(
                possible_combinations, guess_arr, total_correct.item(), positions_correct.item()
            )
            possible_combinations = possible_combinations[mask]
            
            attempt += 1

# ======================================================================================
# Python 侧的包装与工作流
# ======================================================================================

app = typer.Typer()

def get_device(disable_gpu: bool) -> str:
    """根据用户选择和可用性决定计算设备"""
    if disable_gpu:
        return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'

@app.command()
def check(
    secret: str = typer.Argument(..., help="谜底数字"),
    disable_gpu: bool = typer.Option(False, "--disable-gpu", help="禁用 GPU 加速")
):
    """
    检查模式：根据给定的谜底，检查猜测的数字。
    """
    device = get_device(disable_gpu)
    print(f"进入检查模式，谜底为: {secret} (使用设备: {device})")
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
            result = TorchNumleSolver.check(secret, guess, device=device)
            print(f"结果: 包含数字: {result['total_digits']}, 位置正确: {result['correct_positions']}")
        except ValueError as e:
            print(f"错误: {e}")

@app.command()
def solve(
    length: int = typer.Option(5, "--length", "-l", help="数字长度"),
    disable_gpu: bool = typer.Option(False, "--disable-gpu", help="禁用 GPU 加速")
):
    """
    求解模式：交互式求解模式。
    """
    device = get_device(disable_gpu)
    solver = TorchNumleSolver(length, device=device)
    solver.solve_interactive()

@app.command()
def auto_solve(
    secret: str = typer.Argument(None, help="目标数字 (可选，若不提供则随机生成)"),
    length: int = typer.Option(5, "--length", "-l", help="数字长度 (当secret未提供时生效)"),
    disable_gpu: bool = typer.Option(False, "--disable-gpu", help="禁用 GPU 加速")
):
    """
    自动求解模式：给定一个目标数，自动调用求解和检查模式，并且展示过程。
    """
    device = get_device(disable_gpu)
    
    try:
        if secret is None:
            print(f"未提供目标数字，将随机生成一个长度为 {length} 的数字。")
            solver = TorchNumleSolver(length, device=device)
            all_secrets = solver.all_combinations
            secret_idx = torch.randint(0, len(all_secrets), (1,)).item()
            secret_arr = all_secrets[secret_idx]
            secret = "".join([str(i.item()) for i in secret_arr])
            print(f"已生成谜底: {secret}")
        else:
            length = len(secret)
            if not secret.isdigit() or len(set(secret)) != length:
                raise ValueError("目标数字必须只包含数字，且不能有重复。")
            solver = TorchNumleSolver(length, device=device)
    except ValueError as e:
        print(f"错误: {e}")
        raise typer.Exit(code=1)

    print(f"进入自动求解模式，目标为: {secret} (使用设备: {device})")
    
    solver.reset()
    attempt = 1
    start_time = time.time()
    while True:
        guess = solver.get_next_guess()
        if guess is None:
            print("错误：无法找到下一个猜测，可能存在矛盾或已无可能性。")
            break
        
        print(f"\n第 {attempt} 次猜测: {guess}")
        
        result = TorchNumleSolver.check(secret, guess, device=device)
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
def test(
    length: int = typer.Option(5, "--length", "-l", help="数字长度"),
    disable_gpu: bool = typer.Option(False, "--disable-gpu", help="禁用 GPU 加速"),
    limit: int = typer.Option(None, "--limit", help="限制测试用例数量 (可选)")
):
    """
    测试模式：使用 PyTorch 自动遍历所有可能性。
    """
    device = get_device(disable_gpu)
    solver = TorchNumleSolver(length, device=device)
    
    all_secrets = solver.all_combinations
    if limit is not None and limit < len(all_secrets):
        total_tests = limit
        indices = torch.randperm(len(all_secrets))[:limit]
        all_secrets = all_secrets[indices]
    else:
        total_tests = len(all_secrets)

    print("\n开始自动遍历测试...")
    print(f"数字长度: {length}, 总测试数: {total_tests}, 设备: {device}")

    # 预热
    print("PyTorch 预热中...")
    start_time = time.time()
    _solve_one_secret_torch(solver, all_secrets[0])
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    print(f"预热完成，耗时: {end_time - start_time:.2f} 秒。")

    # 执行主计算
    start_time = time.time()
    
    results = torch.empty((total_tests, 2), dtype=torch.int32, device='cpu')
    for i in tqdm(range(total_tests), desc="测试进度", unit="题"):
        is_success, attempts = _solve_one_secret_torch(solver, all_secrets[i])
        results[i, 0] = is_success
        results[i, 1] = attempts

    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    print("计算完成。")

    # 结果统计
    # 结果统计 (使用 PyTorch)
    success_mask = results[:, 0] == 1
    success_count = torch.sum(success_mask).item()
    
    print(f"\n测试完成！结果:")
    print(f"测试总数: {total_tests}")
    print(f"成功次数: {success_count}")
    
    if total_tests > 0:
        print(f"成功率: {success_count / total_tests * 100:.2f}%")
        
    if success_count > 0:
        successful_attempts = results[success_mask, 1]
        total_attempts = torch.sum(successful_attempts).item()
        max_attempts = torch.max(successful_attempts).item()
        print(f"平均猜测次数: {total_attempts / success_count:.2f}")
        print(f"最高猜测次数: {max_attempts}")
        
    print(f"总耗时: {total_time:.2f}秒")
    if total_tests > 0:
        print(f"平均每个测试耗时: {total_time / total_tests:.4f}秒")

if __name__ == "__main__":
    app()