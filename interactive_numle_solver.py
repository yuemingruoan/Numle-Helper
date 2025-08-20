import numpy as np
import numba
import itertools
import time
import threading
from tqdm import tqdm

# ======================================================================================
# Numba JIT 优化的核心计算函数
# ======================================================================================

@numba.njit(cache=True, fastmath=True)
def _check_nb(s, g):
    """Numba JIT: 检查 secret 和 guess"""
    correct_positions = 0
    for i in range(s.shape[0]):
        if s[i] == g[i]:
            correct_positions += 1

    secret_counts = np.zeros(10, dtype=np.int32)
    for i in range(s.shape[0]):
        secret_counts[s[i]] += 1
        
    guess_counts = np.zeros(10, dtype=np.int32)
    for i in range(g.shape[0]):
        guess_counts[g[i]] += 1
        
    correct_digits = 0
    for i in range(10):
        correct_digits += min(secret_counts[i], guess_counts[i])
        
    return correct_digits, correct_positions

@numba.njit(cache=True, fastmath=True)
def _filter_combinations_nb_inplace(mask, combinations, guess_arr, total_correct, positions_correct):
    """Numba JIT: 原地过滤不满足条件的组合 (修改 mask)"""
    n_combinations, L = combinations.shape
    
    guess_counts = np.zeros(10, dtype=np.int32)
    for i in range(L):
        guess_counts[guess_arr[i]] += 1
        
    comb_counts = np.zeros(10, dtype=np.int32)

    for i in range(n_combinations):
        if not mask[i]:
            continue

        comb = combinations[i]
        
        # 1. 检查位置正确数
        pos_correct_count = 0
        for j in range(L):
            if comb[j] == guess_arr[j]:
                pos_correct_count += 1
        
        if pos_correct_count != positions_correct:
            mask[i] = False
            continue

        # 2. 检查数字正确数
        for k in range(10): comb_counts[k] = 0
        for j in range(L):
            comb_counts[comb[j]] += 1
            
        total_correct_count = 0
        for j in range(10):
            total_correct_count += min(comb_counts[j], guess_counts[j])

        if total_correct_count != total_correct:
            mask[i] = False

@numba.njit(cache=True, fastmath=True)
def _filter_combinations_nb(combinations, guess_arr, total_correct, positions_correct):
    """Numba JIT: 过滤不满足条件的组合 (通过调用原地修改版本实现)"""
    n_combinations = combinations.shape[0]
    mask = np.ones(n_combinations, dtype=np.bool_)
    _filter_combinations_nb_inplace(mask, combinations, guess_arr, total_correct, positions_correct)
    return mask

@numba.njit(cache=True, fastmath=True)
def _get_best_guess_idx_from_candidates_nb(candidates, L):
    """
    Numba JIT: 从一组候选中计算熵并找到最佳猜测。
    - 用于从一个已经是子集的数组中（无掩码）找最佳猜测。
    - 返回熵最高的候选在 candidates 数组中的索引。
    """
    N = candidates.shape[0]
    if N == 0:
        return -1
    if N == 1:
        return 0

    # 1. 统计数字分布
    counts = np.zeros((L, 10), dtype=np.int32)
    for i in range(N):
        comb = candidates[i]
        for j in range(L):
            counts[j, comb[j]] += 1

    # 2. 计算频率和信息贡献
    freq = counts / float(N)
    contrib = np.zeros_like(freq, dtype=np.float32)
    for i in range(L):
        for j in range(10):
            if freq[i, j] > 0:
                contrib[i, j] = -freq[i, j] * np.log2(freq[i, j])

    # 3. 在一次遍历中计算熵并找到最佳猜测
    best_idx = -1
    max_entropy = -1.0
    for i in range(N):
        e = 0.0
        comb = candidates[i]
        for j in range(L):
            e += contrib[j, comb[j]]
        
        if e > max_entropy:
            max_entropy = e
            best_idx = i
            
    return best_idx

@numba.njit(cache=True, fastmath=True)
def _get_best_guess_from_mask_nb(candidates, mask, L):
    """
    Numba JIT: 高效地从掩码中计算熵并找到最佳猜测。
    - 合并了熵计算和最大熵搜索，以减少对所有组合的循环次数。
    - 仅对掩码指定的候选进行操作。
    - 返回熵最高的候选的索引。
    """
    N = np.sum(mask)
    if N == 0:
        return -1 # 没有找到
    
    # 如果只剩一个可能性，直接返回那个可能性的索引，无需计算熵
    if N == 1:
        for i in range(candidates.shape[0]):
            if mask[i]:
                return i
        return -1 # 理论上不会发生

    # 1. 统计数字分布
    counts = np.zeros((L, 10), dtype=np.int32)
    for i in range(candidates.shape[0]):
        if mask[i]:
            comb = candidates[i]
            for j in range(L):
                counts[j, comb[j]] += 1

    # 2. 计算频率和信息贡献
    freq = counts / float(N)
    contrib = np.zeros_like(freq, dtype=np.float32)
    for i in range(L):
        for j in range(10):
            if freq[i, j] > 0:
                contrib[i, j] = -freq[i, j] * np.log2(freq[i, j])

    # 3. 在一次遍历中计算熵并找到最佳猜测
    # 我们的猜测本身也必须是众多可能性之一
    best_idx = -1
    max_entropy = -1.0
    for i in range(candidates.shape[0]):
        if mask[i]:
            e = 0.0
            comb = candidates[i]
            for j in range(L):
                e += contrib[j, comb[j]]
            
            if e > max_entropy:
                max_entropy = e
                best_idx = i
                
    return best_idx

# ======================================================================================
# Numba JIT 优化的核心求解器 (从 main.py 移入)
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

def _test_all_nb_single_thread(all_combinations, results, first_guess_arr, progress_callback=None):
    """
    测试所有组合 (单线程)，支持进度更新
    """
    n_tests = len(all_combinations)
    for i in range(n_tests):
        secret = all_combinations[i]
        is_success, attempts = _solve_one_secret_nb(all_combinations, secret, first_guess_arr)
        results[i, 0] = is_success
        results[i, 1] = attempts
        
        # 如果提供了进度回调函数，则调用它
        if progress_callback is not None:
            progress_callback(i + 1)

# ======================================================================================
# 主类
# ======================================================================================

class InteractiveNumleSolver:
    def __init__(self, digit_length=5):
        if not isinstance(digit_length, int) or not 1 <= digit_length <= 10:
            raise ValueError("数字长度必须是 1 到 10 之间的整数")
        self.digit_length = digit_length
        self.all_combinations = self.generate_all_combinations()
        self.possible_combinations = self.all_combinations.copy()
        
    def generate_all_combinations(self):
        """生成所有不含重复数字的可能组合"""
        digits = np.arange(10, dtype=np.uint8)
        all_perms = list(itertools.permutations(digits, self.digit_length))
        return np.array(all_perms, dtype=np.uint8)
    
    def get_next_guess(self):
        """获取下一个猜测（调用 Numba JIT 核心）"""
        if len(self.possible_combinations) == 0:
            return None

        candidates = self.possible_combinations
        
        # 调用新的 Numba JIT 函数获取最佳猜测的索引
        best_idx = _get_best_guess_idx_from_candidates_nb(candidates, self.digit_length)
        if best_idx == -1:
            return None

        return ''.join(map(str, candidates[best_idx]))
        
    @staticmethod
    def check(secret, guess):
        """检查模式（调用 Numba JIT 核心）"""
        if len(secret) != len(guess):
            raise ValueError("秘密数字和猜测数字长度不一致")
            
        s = np.frombuffer(secret.encode('ascii'), dtype=np.uint8) - 48
        g = np.frombuffer(guess.encode('ascii'), dtype=np.uint8) - 48
        
        total_digits, correct_positions = _check_nb(s, g)

        return {
            "total_digits": total_digits,
            "correct_positions": correct_positions
        }
    
    def update_possible_combinations(self, guess, total_correct, positions_correct):
        """更新可能数字组合（调用 Numba JIT 核心）"""
        if len(self.possible_combinations) == 0:
            return

        guess_arr = np.frombuffer(guess.encode('ascii'), dtype=np.uint8) - 48
        
        # 调用重构后的 Numba JIT 函数进行过滤
        mask = _filter_combinations_nb(
            self.possible_combinations, guess_arr, total_correct, positions_correct
        )
        
        self.possible_combinations = self.possible_combinations[mask]
    
    def reset(self):
        """重置求解器状态"""
        self.possible_combinations = self.all_combinations.copy()

    def solve_interactive(self):
        """交互式求解主函数"""
        self.reset()
        print(f"Numle 求解器已启动（{self.digit_length}位数字）")
        print("提示：游戏反馈应包含两个数字：")
        print("  1. 总共包含的数字数量（无论位置）")
        print("  2. 位置也正确的数字数量")
        
        attempt = 1
        while True:
            guess = self.get_next_guess()
            if guess is None:
                print("无有效数字可猜，可能之前的反馈有误。游戏结束。")
                return
                
            print(f"\n尝试 #{attempt}: 我猜: {guess}")
            
            while True:
                try:
                    feedback_str = input("请输入反馈（格式：包含数字 位置正确）: ").split()
                    if len(feedback_str) != 2:
                        print("格式错误！请输入两个数字（例如：3 1）")
                        continue
                    
                    total_correct = int(feedback_str[0])
                    positions_correct = int(feedback_str[1])

                    if not (0 <= positions_correct <= total_correct <= self.digit_length):
                        print(f"数字范围错误！总包含数应在0-{self.digit_length}之间，位置正确数应在0-总包含数之间。")
                        continue
                        
                    if positions_correct == self.digit_length:
                        print(f"\n成功！数字是：{guess}")
                        return
                        
                    self.update_possible_combinations(guess, total_correct, positions_correct)
                    print(f"剩余可能组合数: {len(self.possible_combinations)}")
                    attempt += 1
                    break

                except ValueError:
                    print("格式错误！请输入两个整数数字（例如：3 1）")
                    continue
    
    def run_benchmark(self, parallel=True):
        """运行基准测试（从 main.py 移入）"""
        all_secrets = self.all_combinations
        total_tests = len(all_secrets)
        length = self.digit_length

        print("\n开始自动遍历测试...")
        print(f"数字长度: {length}, 总测试数: {total_tests}, 并行计算: {'启用' if parallel else '禁用'}")
        if parallel:
            print(f"Numba 并行数量: {numba.get_num_threads()}")

        # Numba JIT 预热
        print("Numba JIT 预热中...")
        start_time = time.time()
        
        # 预热时需要计算首猜
        first_guess_idx = _get_best_guess_idx_from_candidates_nb(all_secrets, length)
        first_guess_arr = all_secrets[first_guess_idx]
        
        # 预热 JIT 函数
        dummy_results = np.empty((1, 2), dtype=np.int32)
        _test_all_nb(all_secrets[:1], dummy_results, first_guess_arr) # 并行预热
        _solve_one_secret_nb(all_secrets, all_secrets[0], first_guess_arr) # 单核求解器预热
        _test_all_nb_single_thread(all_secrets[:1], dummy_results, first_guess_arr, None) # 单线程测试循环预热

        end_time = time.time()
        print(f"预热完成，耗时: {end_time - start_time:.2f} 秒。")
        print(f"预计算出的最佳首次猜测: {''.join(map(str, first_guess_arr))}")

        # --- 执行主计算 ---
        start_time = time.time()
        results = np.empty((total_tests, 2), dtype=np.int32)
        results.fill(-1)
        
        desc = "并行测试进度" if parallel else "单线程测试进度"
        
        if parallel:
            # 并行模式使用原有逻辑
            worker = threading.Thread(target=_test_all_nb, args=(all_secrets, results, first_guess_arr))
            worker.daemon = True
            worker.start()
            
            interrupted = False
            try:
                with tqdm(total=total_tests, desc=desc) as pbar:
                    while worker.is_alive():
                        processed = np.sum(results[:, 1] != -1)
                        pbar.update(processed - pbar.n)
                        time.sleep(0.1)
                    processed = np.sum(results[:, 1] != -1)
                    pbar.update(processed - pbar.n)
            except KeyboardInterrupt:
                interrupted = True
                print("\n\n测试被用户中断。正在处理已完成部分的结果...")
        else:
            # 非并行模式使用回调函数更新进度
            with tqdm(total=total_tests, desc=desc) as pbar:
                def progress_callback(processed):
                    pbar.update(processed - pbar.n)
                
                interrupted = False
                try:
                    _test_all_nb_single_thread(all_secrets, results, first_guess_arr, progress_callback)
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
            return

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

            print("\n猜测次数分布:")
            attempts_distribution = np.bincount(successful_attempts)
            for i, count in enumerate(attempts_distribution):
                if i > 0 and count > 0:
                    percentage = count / success_count * 100
                    print(f"  {i} 次: {count:5d} 个 ({percentage:5.2f}%)")
            
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
        if total_processed > 0 and total_time > 0:
            total_guesses = np.sum(results[processed_mask, 1])
            problems_per_second = total_processed / total_time
            guesses_per_second = total_guesses / total_time
            
            print(f"平均每个测试耗时: {total_time / total_processed:.4f}秒")
            print(f"每秒解题数 (TPS): {problems_per_second:.2f}")
            print(f"每秒猜测数 (GPS): {guesses_per_second:.2f}")