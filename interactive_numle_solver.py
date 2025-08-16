import numpy as np
import numba
import itertools

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
def _calculate_entropies_nb(candidates, L):
    """Numba JIT: 计算所有候选组合的信息熵"""
    N = len(candidates)
    
    # 统计每一列(位置)的数字分布
    counts = np.zeros((L, 10), dtype=np.int32)
    for i in range(L):
        for j in range(N):
            counts[i, candidates[j, i]] += 1

    # 概率与信息贡献表
    freq = counts / float(N)
    contrib = np.zeros_like(freq, dtype=np.float32)
    for i in range(L):
        for j in range(10):
            if freq[i, j] > 0:
                contrib[i, j] = -freq[i, j] * np.log2(freq[i, j])

    # 对所有候选计算熵
    entropies = np.zeros(N, dtype=np.float32)
    for i in range(N):
        e = 0.0
        for j in range(L):
            e += contrib[j, candidates[i, j]]
        entropies[i] = e
        
    return entropies

@numba.njit(cache=True, fastmath=True)
def _filter_combinations_nb(combinations, guess_arr, total_correct, positions_correct):
    """Numba JIT: 过滤不满足条件的组合"""
    n_combinations, L = combinations.shape
    
    # 预分配掩码数组
    mask = np.ones(n_combinations, dtype=np.bool_)
    
    # 临时计数数组
    comb_counts = np.zeros(10, dtype=np.int32)
    guess_counts = np.zeros(10, dtype=np.int32)
    for i in range(guess_arr.shape[0]):
        guess_counts[guess_arr[i]] += 1

    for i in range(n_combinations):
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
        # 重置并计算当前组合的数字分布
        for k in range(10): comb_counts[k] = 0
        for j in range(L):
            comb_counts[comb[j]] += 1
            
        total_correct_count = 0
        for j in range(10):
            total_correct_count += min(comb_counts[j], guess_counts[j])

        if total_correct_count != total_correct:
            mask[i] = False
            
    # 返回最终的布尔掩码
    return mask


@numba.njit(cache=True, fastmath=True)
def _calculate_entropies_nb_masked(candidates, mask, L):
    """Numba JIT: 根据掩码计算候选组合的信息熵"""
    N = np.sum(mask)
    if N == 0:
        return np.zeros(len(candidates), dtype=np.float32)

    counts = np.zeros((L, 10), dtype=np.int32)
    for i in range(len(candidates)):
        if mask[i]:
            comb = candidates[i]
            for j in range(L):
                counts[j, comb[j]] += 1

    freq = counts / float(N)
    contrib = np.zeros_like(freq, dtype=np.float32)
    for i in range(L):
        for j in range(10):
            if freq[i, j] > 0:
                contrib[i, j] = -freq[i, j] * np.log2(freq[i, j])

    entropies = np.zeros(len(candidates), dtype=np.float32)
    for i in range(len(candidates)):
        if mask[i]:
            e = 0.0
            comb = candidates[i]
            for j in range(L):
                e += contrib[j, comb[j]]
            entropies[i] = e
            
    return entropies

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

# ======================================================================================
# 主类
# ======================================================================================

class InteractiveNumleSolver:
    def __init__(self, digit_length=5):
        self.digit_length = digit_length
        self.possible_combinations = np.empty((0, self.digit_length), dtype=np.uint8)
        self.all_combinations = self.generate_all_combinations()
        
    def generate_all_combinations(self):
        """生成所有不含重复数字的可能组合"""
        if self.digit_length > 10:
            raise ValueError("数字长度不能超过10，因为数字不能重复")
        
        # 使用itertools.permutations生成所有不重复的组合
        digits = np.arange(10, dtype=np.uint8)
        all_perms = list(itertools.permutations(digits, self.digit_length))
        return np.array(all_perms, dtype=np.uint8)
    
    def get_next_guess(self):
        """获取下一个猜测（调用 Numba JIT 核心）"""
        if len(self.possible_combinations) == 0:
            self.possible_combinations = self.all_combinations
            
        if len(self.possible_combinations) == 0:
            return None

        candidates = self.possible_combinations
        
        # 调用 Numba JIT 函数计算熵
        entropies = _calculate_entropies_nb(candidates, self.digit_length)

        best_idx = int(np.argmax(entropies))
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
        
        # 调用 Numba JIT 函数进行过滤
        mask = _filter_combinations_nb(
            self.possible_combinations, guess_arr, total_correct, positions_correct
        )
        
        self.possible_combinations = self.possible_combinations[mask]
    
    def solve_interactive(self, max_attempts=6):
        """交互式求解主函数"""
        print(f"Numle 求解器已启动（{self.digit_length}位数字）")
        print("提示：游戏反馈应包含两个数字：")
        print("  1. 总共包含的数字数量（无论位置）")
        print("  2. 位置也正确的数字数量")
        
        attempt = 1
        while True:
            if len(self.possible_combinations) == 0:
                self.possible_combinations = self.all_combinations
                
            guess = self.get_next_guess()
            if guess is None:
                print("无有效数字可猜，游戏结束")
                return False
                
            print(f"\n尝试 #{attempt}: 我猜: {guess}")
            
            while True:
                feedback = input("请输入反馈（格式：包含数字数量 位置正确数量）: ").split()
                
                if len(feedback) != 2:
                    print("格式错误！请输入两个数字（例如：3 1）")
                    continue
                    
                try:
                    total_correct = int(feedback[0])
                    positions_correct = int(feedback[1])
                except ValueError:
                    print("格式错误！请输入两个整数数字（例如：3 1）")
                    continue
                    
                if total_correct < 0 or positions_correct < 0 or total_correct > self.digit_length or positions_correct > total_correct:
                    print(f"数字范围错误！总包含数应在0-{self.digit_length}之间，位置正确数应在0-总包含数之间")
                    continue
                    
                if positions_correct == self.digit_length:
                    print(f"\n成功！数字是：{guess}")
                    return True
                    
                self.update_possible_combinations(guess, total_correct, positions_correct)
                print(f"剩余可能组合数: {len(self.possible_combinations)}")
                attempt += 1
                break