import torch
import torch.nn.functional as F
import itertools

# ======================================================================================
# 主类
# ======================================================================================

class TorchNumleSolver:
    @staticmethod
    @torch.jit.script
    def _check_torch(s: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """PyTorch JIT: 检查 secret 和 guess"""
        # 1. 检查位置正确数
        correct_positions = (s == g).sum()

        # 2. 检查数字正确数 (使用 one-hot, 对 JIT/GPU 更友好)
        s_one_hot = F.one_hot(s, num_classes=10)
        g_one_hot = F.one_hot(g, num_classes=10)
        s_counts = s_one_hot.sum(dim=0)
        g_counts = g_one_hot.sum(dim=0)
        correct_digits = torch.min(s_counts, g_counts).sum()
        
        return correct_digits, correct_positions

    @staticmethod
    @torch.jit.script
    def _calculate_entropies_torch(candidates: torch.Tensor) -> torch.Tensor:
        """PyTorch JIT: 计算所有候选组合的信息熵"""
        N, L = candidates.shape
        device = candidates.device

        # 1. 统计每一列(位置)的数字分布
        one_hot_candidates = F.one_hot(candidates, num_classes=10).to(torch.float32)
        counts = one_hot_candidates.sum(dim=0)

        # 2. 计算概率与信息贡献
        freq = counts / float(N)
        non_zero_freq = freq > 0
        log_freq = torch.zeros_like(freq)
        log_freq[non_zero_freq] = torch.log2(freq[non_zero_freq])
        contrib = -freq * log_freq  # Shape: (L, 10)

        # 3. 对所有候选计算熵 (向量化版本)
        # 扩展 contrib 以匹配 candidates 的批次大小
        # contrib shape: (L, 10) -> (1, L, 10) -> (N, L, 10)
        contrib_expanded = contrib.unsqueeze(0).expand(N, -1, -1)
        
        # 扩展 candidates 以用作 gather 的索引
        # candidates shape: (N, L) -> (N, L, 1)
        candidates_expanded = candidates.unsqueeze(-1)

        # 使用 gather 收集每个候选组合的熵贡献
        # gathered_entropies shape: (N, L, 1)
        gathered_entropies = torch.gather(contrib_expanded, 2, candidates_expanded)

        # 求和得到最终熵
        # entropies shape: (N,)
        entropies = gathered_entropies.squeeze(-1).sum(dim=1)
        
        return entropies

    @staticmethod
    @torch.jit.script
    def _filter_combinations_torch(combinations: torch.Tensor, guess_arr: torch.Tensor, total_correct: int, positions_correct: int) -> torch.Tensor:
        """PyTorch JIT: 过滤不满足条件的组合"""
        # 1. 检查位置正确数
        pos_correct_counts = (combinations == guess_arr).sum(dim=1)
        pos_mask = (pos_correct_counts == positions_correct)
        
        # 如果掩码已经全为 False，提前返回
        if not torch.any(pos_mask):
            return pos_mask

        # 2. 检查数字正确数
        active_combinations = combinations[pos_mask]
        
        if active_combinations.numel() == 0:
            return pos_mask

        # 使用 one-hot 批量计算 (比 bincount 更适合 JIT)
        active_one_hot = F.one_hot(active_combinations, num_classes=10)
        comb_counts = active_one_hot.sum(dim=1)
        
        guess_one_hot = F.one_hot(guess_arr, num_classes=10)
        guess_counts = guess_one_hot.sum(dim=0)

        total_correct_counts = torch.min(comb_counts, guess_counts).sum(dim=1)
        
        # 计算第二阶段的掩码
        total_mask = (total_correct_counts == total_correct)
        
        # 将第二阶段的结果更新回原始掩码
        final_mask = pos_mask.clone()
        final_mask[pos_mask] = total_mask
        
        return final_mask

    @staticmethod
    @torch.jit.script
    def _check_torch_batched(s: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """PyTorch JIT: 批量检查 secrets 和 guesses
        s: (B, L)
        g: (B, L)
        """
        B, L = s.shape
        
        # 1. 批量检查位置正确数
        correct_positions = (s == g).sum(dim=1)

        # 2. 批量检查数字正确数
        s_one_hot = F.one_hot(s, num_classes=10)  # (B, L, 10)
        g_one_hot = F.one_hot(g, num_classes=10)  # (B, L, 10)
        s_counts = s_one_hot.sum(dim=1)  # (B, 10)
        g_counts = g_one_hot.sum(dim=1)  # (B, 10)
        correct_digits = torch.min(s_counts, g_counts).sum(dim=1)
        
        return correct_digits, correct_positions

    @staticmethod
    @torch.jit.script
    def _filter_combinations_torch_batched(
        combinations: torch.Tensor, guess_arr: torch.Tensor,
        total_correct: torch.Tensor, positions_correct: torch.Tensor
    ) -> torch.Tensor:
        """PyTorch JIT: 批量过滤不满足条件的组合
        combinations: (B, N, L)
        guess_arr: (B, L)
        total_correct: (B,)
        positions_correct: (B,)
        """
        B, N, L = combinations.shape
        
        # 1. 批量检查位置正确数
        # guess_arr (B, L) -> (B, 1, L)
        # combinations (B, N, L) vs guess_arr (B, 1, L) -> (B, N, L)
        pos_correct_counts = (combinations == guess_arr.unsqueeze(1)).sum(dim=2) # (B, N)
        pos_mask = (pos_correct_counts == positions_correct.unsqueeze(1)) # (B, N)

        # 2. 批量检查数字正确数
        # one-hot 转换
        comb_one_hot = F.one_hot(combinations, num_classes=10) # (B, N, L, 10)
        comb_counts = comb_one_hot.sum(dim=2) # (B, N, 10)

        guess_one_hot = F.one_hot(guess_arr, num_classes=10) # (B, L, 10)
        guess_counts = guess_one_hot.sum(dim=1) # (B, 10)

        # guess_counts (B, 10) -> (B, 1, 10)
        total_correct_counts = torch.min(comb_counts, guess_counts.unsqueeze(1)).sum(dim=2) # (B, N)
        total_mask = (total_correct_counts == total_correct.unsqueeze(1)) # (B, N)
        
        # 合并两个掩码
        return pos_mask & total_mask

    def __init__(self, digit_length=5, device='cpu'):
        self.digit_length = digit_length
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # 生成所有组合并移动到指定设备
        self.all_combinations = self.generate_all_combinations().to(self.device)
        self.possible_combinations = self.all_combinations.clone()

    def generate_all_combinations(self):
        """生成所有不含重复数字的可能组合"""
        if self.digit_length > 10:
            raise ValueError("数字长度不能超过10，因为数字不能重复")
        
        digits = range(10)
        all_perms = list(itertools.permutations(digits, self.digit_length))
        return torch.tensor(all_perms, dtype=torch.long)

    def get_next_guess(self):
        """获取下一个猜测（调用 PyTorch 核心）"""
        if len(self.possible_combinations) == 0:
            return None

        candidates = self.possible_combinations
        
        # 调用 PyTorch 函数计算熵
        entropies = self._calculate_entropies_torch(candidates)

        best_idx = torch.argmax(entropies)
        best_guess_tensor = candidates[best_idx]
        return "".join([str(i.item()) for i in best_guess_tensor])
        
    @staticmethod
    def check(secret, guess, device='cpu'):
        """检查模式（调用 PyTorch 核心）"""
        if len(secret) != len(guess):
            raise ValueError("秘密数字和猜测数字长度不一致")
            
        s = torch.tensor([int(d) for d in secret], dtype=torch.long, device=device)
        g = torch.tensor([int(d) for d in guess], dtype=torch.long, device=device)
        
        total_digits, correct_positions = TorchNumleSolver._check_torch(s, g)

        return {
            "total_digits": total_digits.item(),
            "correct_positions": correct_positions.item()
        }
    
    def update_possible_combinations(self, guess, total_correct, positions_correct):
        """更新可能数字组合（调用 PyTorch 核心）"""
        if len(self.possible_combinations) == 0:
            return

        guess_arr = torch.tensor([int(d) for d in guess], dtype=torch.long, device=self.device)
        
        # 调用 PyTorch 函数进行过滤
        mask = self._filter_combinations_torch(
            self.possible_combinations, guess_arr, total_correct, positions_correct
        )
        
        self.possible_combinations = self.possible_combinations[mask]
    
    def reset(self):
        """重置求解器状态"""
        self.possible_combinations = self.all_combinations.clone()

    def solve_interactive(self, max_attempts=10):
        """交互式求解主函数"""
        self.reset()
        print(f"Numle 求解器已启动（{self.digit_length}位数字, 使用 {self.device}）")
        print("提示：游戏反馈应包含两个数字：")
        print("  1. 总共包含的数字数量（无论位置）")
        print("  2. 位置也正确的数字数量")
        
        attempt = 1
        while attempt <= max_attempts:
            guess = self.get_next_guess()
            if guess is None:
                print("无有效数字可猜，可能之前的反馈有误。")
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
                    
                if not (0 <= total_correct <= self.digit_length and 0 <= positions_correct <= total_correct):
                    print(f"数字范围错误！总包含数应在0-{self.digit_length}之间，位置正确数应在0-总包含数之间")
                    continue
                    
                if positions_correct == self.digit_length:
                    print(f"\n成功！数字是：{guess}")
                    return True
                    
                self.update_possible_combinations(guess, total_correct, positions_correct)
                print(f"剩余可能组合数: {len(self.possible_combinations)}")
                break
            
            attempt += 1
        
        print(f"\n超过最大尝试次数 ({max_attempts})。")
        if len(self.possible_combinations) == 1:
            final_answer = "".join([str(i.item()) for i in self.possible_combinations[0]])
            print(f"唯一剩下的答案是: {final_answer}")
        elif 0 < len(self.possible_combinations) <= 10:
            print("剩余的可能性:")
            for comb in self.possible_combinations:
                print("".join([str(i.item()) for i in comb]))
        return False