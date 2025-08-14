import numpy as np

class InteractiveNumleSolver:
    def __init__(self, digit_length=5):
        self.digit_length = digit_length
        self.possible_combinations = np.empty((0, self.digit_length), dtype=np.uint8)
        self.all_combinations = self.generate_all_combinations()
        
    def generate_all_combinations(self):
        """生成所有可能的数字组合"""
        digits = np.arange(10, dtype=np.uint8)
        grids = np.meshgrid(*([digits]*self.digit_length))
        return np.stack(grids, axis=-1).reshape(-1, self.digit_length)
    
    def get_next_guess(self):
        """获取下一个猜测"""
        if len(self.possible_combinations) == 0:
            self.possible_combinations = self.all_combinations.copy()
            
        if len(self.possible_combinations) == 0:
            return None
            
        # 计算每个位置数字频率（优化后的向量化实现）
        sample_size = min(1000, len(self.possible_combinations))
        candidates = self.possible_combinations[:sample_size]
        
        # 计算每个候选的熵
        entropies = np.zeros(len(candidates))
        for i in range(self.digit_length):
            # 计算当前位的数字频率
            unique, counts = np.unique(self.possible_combinations[:, i], return_counts=True)
            freq = np.zeros(10, dtype=np.float64)
            freq[unique] = counts / len(self.possible_combinations)
            
            # 计算当前位对熵的贡献
            p = freq[candidates[:, i]]
            entropies += np.where(p > 0, -p * np.log2(p), 0)
        
        best_idx = np.argmax(entropies)
        return ''.join(map(str, candidates[best_idx]))
        
    @staticmethod
    def check(secret, guess):
        """检查模式"""
        secret = np.array(list(map(int, secret)))
        guess = np.array(list(map(int, guess)))
        
        if len(secret) != len(guess):
            raise ValueError("秘密数字和猜测数字长度不一致")
            
        correct_positions = (secret == guess).sum()
        
        secret_counts = np.bincount(secret, minlength=10)
        guess_counts = np.bincount(guess, minlength=10)
        correct_digits = np.minimum(secret_counts, guess_counts).sum()
        
        return {
            "total_digits": correct_digits,
            "correct_positions": correct_positions
        }
    
    def update_possible_combinations(self, guess, total_correct, positions_correct):
        """更新可能数字组合"""
        if len(self.possible_combinations) == 0:
            return
            
        guess_arr = np.array(list(map(int, guess)), dtype=np.uint8)
        
        correct_positions = (self.possible_combinations == guess_arr).sum(axis=1)
        
        secret_counts = np.zeros((len(self.possible_combinations), 10), dtype=np.uint8)
        for i in range(10):
            secret_counts[:, i] = (self.possible_combinations == i).sum(axis=1)
            
        guess_counts = np.bincount(guess_arr, minlength=10)
        total_digits = np.minimum(secret_counts, guess_counts).sum(axis=1)
        
        mask = (total_digits == total_correct) & (correct_positions == positions_correct)
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
                self.possible_combinations = self.all_combinations.copy()
                
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