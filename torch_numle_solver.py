import torch
import torch.nn.functional as F
import itertools
import numpy as np

# ======================================================================================
# 主类
# ======================================================================================

class TorchNumleSolver:
    @staticmethod
    def _check_torch(s, g):
        """PyTorch: 检查 secret 和 guess"""
        # 确保输入是 tensor
        if not isinstance(s, torch.Tensor):
            s = torch.from_numpy(s)
        if not isinstance(g, torch.Tensor):
            g = torch.from_numpy(g)
        
        device = s.device
        
        # 1. 检查位置正确数
        correct_positions = (s == g).sum()

        # 2. 检查数字正确数
        s_counts = torch.bincount(s, minlength=10)
        g_counts = torch.bincount(g, minlength=10)
        correct_digits = torch.min(s_counts, g_counts).sum()
        
        return correct_digits, correct_positions

    @staticmethod
    def _calculate_entropies_torch(candidates):
        """PyTorch: 计算所有候选组合的信息熵"""
        N, L = candidates.shape
        device = candidates.device

        # 1. 统计每一列(位置)的数字分布
        # 使用 one-hot 编码来计数
        one_hot_candidates = F.one_hot(candidates, num_classes=10).float() # (N, L, 10)
        counts = one_hot_candidates.sum(dim=0) # (L, 10)

        # 2. 计算概率与信息贡献
        freq = counts / N
        # 使用 where 避免 log(0)
        log_freq = torch.where(freq > 0, torch.log2(freq), torch.tensor(0.0, device=device))
        contrib = -freq * log_freq # (L, 10)

        # 3. 对所有候选计算熵
        # 使用 gather 从贡献表中提取每个候选组合对应位置和数字的熵贡献
        # contrib.unsqueeze(0) -> (1, L, 10)
        # candidates.unsqueeze(-1) -> (N, L, 1)
        # gather 结果 -> (N, L, 1), squeeze(-1) -> (N, L)
        entropies = contrib.gather(1, candidates.T).sum(dim=0)
        
        return entropies

    @staticmethod
    def _filter_combinations_torch(combinations, guess_arr, total_correct, positions_correct):
        """PyTorch: 过滤不满足条件的组合"""
        N, L = combinations.shape
        device = combinations.device

        # 1. 检查位置正确数
        # (N, L) vs (L,) -> 广播 -> (N, L)
        pos_correct_counts = (combinations == guess_arr).sum(dim=1)
        mask = (pos_correct_counts == positions_correct)
        
        # 如果掩码已经全为 False，提前返回
        if not torch.any(mask):
            return mask

        # 2. 检查数字正确数
        # 只对掩码为 True 的组合进行计算
        active_combinations = combinations[mask]
        
        # 使用 bincount 批量计算
        # 先将 active_combinations 转换为 one-hot
        active_one_hot = F.one_hot(active_combinations, num_classes=10) # (n_active, L, 10)
        comb_counts = active_one_hot.sum(dim=1) # (n_active, 10)
        
        guess_counts = torch.bincount(guess_arr, minlength=10) # (10,)

        total_correct_counts = torch.min(comb_counts, guess_counts).sum(dim=1)
        
        # 更新掩码
        mask[mask] = (total_correct_counts == total_correct)
        
        return mask

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
        return ''.join(map(str, best_guess_tensor.cpu().numpy()))
        
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
            final_answer = ''.join(map(str, self.possible_combinations[0].cpu().numpy()))
            print(f"唯一剩下的答案是: {final_answer}")
        elif 0 < len(self.possible_combinations) <= 10:
            print("剩余的可能性:")
            for comb in self.possible_combinations:
                print(''.join(map(str, comb.cpu().numpy())))
        return False