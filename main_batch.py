import typer
import torch
import time
from tqdm import tqdm
from torch_numle_solver import TorchNumleSolver

app = typer.Typer()

def get_device(disable_gpu: bool) -> str:
    """根据用户选择和可用性决定计算设备"""
    if disable_gpu:
        return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'

@app.command()
def test(
    length: int = typer.Option(5, "--length", "-l", help="数字长度"),
    disable_gpu: bool = typer.Option(False, "--disable-gpu", help="禁用 GPU 加速"),
    limit: int = typer.Option(None, "--limit", help="限制测试用例数量 (可选)")
):
    """
    测试模式：使用 PyTorch 批量并行计算。
    """
    device = get_device(disable_gpu)
    solver = TorchNumleSolver(length, device=device)
    
    all_combinations = solver.all_combinations
    secrets_to_solve = all_combinations
    
    if limit is not None and limit < len(secrets_to_solve):
        total_tests = limit
        indices = torch.randperm(len(secrets_to_solve))[:limit]
        secrets_to_solve = secrets_to_solve[indices]
    else:
        total_tests = len(secrets_to_solve)

    print("\n开始批量并行测试...")
    print(f"数字长度: {length}, 总测试数: {total_tests}, 设备: {device}")

    start_time = time.time()

    # --- 批量求解器核心逻辑 ---
    with torch.no_grad():
        B = total_tests  # Batch size is the total number of tests
        N, L = all_combinations.shape

        # 为批次中的每个谜题维护一个候选组合的掩码
        # Shape: (B, N)
        possible_masks = torch.ones(B, N, dtype=torch.bool, device=device)
        
        # 记录每个谜题的尝试次数和最终结果
        attempts = torch.zeros(B, dtype=torch.int32, device='cpu')
        final_attempts = torch.zeros(B, dtype=torch.int32, device='cpu')
        
        # 记录每个谜题是否已解决
        solved_mask = torch.zeros(B, dtype=torch.bool, device=device)
        
        # 主循环
        pbar = tqdm(total=B, desc="解题进度")
        while not torch.all(solved_mask):
            # 找到所有未解决的谜题
            unsolved_indices = torch.where(~solved_mask)[0]
            
            # 对所有未解决的谜题增加尝试次数
            attempts[unsolved_indices] += 1
            
            # --- 选择猜测 ---
            # 简化策略：在所有谜题的剩余候选池的并集中，找到一个全局最佳猜测
            # 1. 获取所有未解决问题的候选池的联合
            # (num_unsolved, N) -> (1, N) -> (N,)
            union_mask = torch.any(possible_masks[unsolved_indices], dim=0)
            candidate_pool = all_combinations[union_mask]

            if len(candidate_pool) == 0:
                print("\n警告：候选池为空，部分谜题可能无解或存在逻辑错误。")
                break

            # 2. 在这个联合池中计算熵，找到最佳猜测
            entropies = solver._calculate_entropies_torch(candidate_pool)
            best_idx = torch.argmax(entropies)
            guess_arr = candidate_pool[best_idx] # Shape: (L,)

            # --- 检查 & 过滤 ---
            current_secrets = secrets_to_solve[unsolved_indices]
            
            # 批量检查
            total_correct, positions_correct = solver._check_torch_batched(current_secrets, guess_arr.unsqueeze(0))
            
            # 检查是否有新解决的谜题
            newly_solved_sub_mask = (positions_correct == L)
            if torch.any(newly_solved_sub_mask):
                newly_solved_global_indices = unsolved_indices[newly_solved_sub_mask]
                solved_mask[newly_solved_global_indices] = True
                final_attempts[newly_solved_global_indices] = attempts[newly_solved_global_indices]
                pbar.update(len(newly_solved_global_indices))

            # 对于那些还没解出来的，继续过滤
            not_yet_solved_sub_mask = ~newly_solved_sub_mask
            if not torch.any(not_yet_solved_sub_mask):
                continue # 如果所有都解出来了，跳过过滤

            # 准备批量过滤
            indices_to_filter = unsolved_indices[not_yet_solved_sub_mask]
            combinations_to_filter = all_combinations.unsqueeze(0).expand(len(indices_to_filter), -1, -1)
            
            # 批量过滤
            filter_mask = solver._filter_combinations_torch_batched(
                combinations_to_filter,
                guess_arr.unsqueeze(0).expand(len(indices_to_filter), -1),
                total_correct[not_yet_solved_sub_mask],
                positions_correct[not_yet_solved_sub_mask]
            )
            
            # 更新主掩码
            possible_masks[indices_to_filter] &= filter_mask

    pbar.close()
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    print("计算完成。")

    # --- 结果统计 ---
    success_count = torch.sum(solved_mask).item()
    
    print(f"\n测试完成！结果:")
    print(f"测试总数: {total_tests}")
    print(f"成功次数: {success_count}")
    
    if total_tests > 0:
        print(f"成功率: {success_count / total_tests * 100:.2f}%")
        
    if success_count > 0:
        successful_attempts = final_attempts[final_attempts > 0]
        total_attempts_sum = torch.sum(successful_attempts).item()
        max_attempts = torch.max(successful_attempts).item()
        print(f"平均猜测次数: {total_attempts_sum / success_count:.2f}")
        print(f"最高猜测次数: {max_attempts}")
        
    print(f"总耗时: {total_time:.2f}秒")
    if total_tests > 0:
        print(f"平均每个测试耗时: {total_time / total_tests:.4f}秒")

if __name__ == "__main__":
    app()