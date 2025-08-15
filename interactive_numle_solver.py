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
        """获取下一个猜测（全向量化评分）"""
        if len(self.possible_combinations) == 0:
            # 直接复用基表，避免不必要的 copy
            self.possible_combinations = self.all_combinations
            
        if len(self.possible_combinations) == 0:
            return None

        # 候选全集
        candidates = self.possible_combinations

        # 统计每一列(位置)的数字分布: counts[pos, digit] in [0..N]
        L = self.digit_length
        N = len(candidates)
        counts = np.empty((L, 10), dtype=np.int32)
        for i in range(L):
            counts[i] = np.bincount(candidates[:, i], minlength=10)

        # 概率与信息贡献表 contrib[pos, digit] = -p*log2(p); p=0 时贡献 0
        freq = counts.astype(np.float32) / float(N)
        contrib = np.zeros_like(freq, dtype=np.float32)
        mask = freq > 0
        contrib[mask] = -freq[mask] * np.log2(freq[mask])

        # 对所有候选一次性取出其在每个位置的贡献并求和
        entropies = contrib[np.arange(L)[:, None], candidates.T].sum(axis=0)

        best_idx = int(np.argmax(entropies))
        return ''.join(map(str, candidates[best_idx]))
        
    @staticmethod
    def check(secret, guess):
        """检查模式（高效字符串转数字与矢量化计数）"""
        # 假定仅含 0-9 字符；若格式不正确可在调用端校验
        s = np.frombuffer(secret.encode('ascii'), dtype=np.uint8) - 48
        g = np.frombuffer(guess.encode('ascii'), dtype=np.uint8) - 48

        if s.size != g.size:
            raise ValueError("秘密数字和猜测数字长度不一致")

        correct_positions = int((s == g).sum())

        secret_counts = np.bincount(s, minlength=10)
        guess_counts = np.bincount(g, minlength=10)
        correct_digits = int(np.minimum(secret_counts, guess_counts).sum())

        return {
            "total_digits": correct_digits,
            "correct_positions": correct_positions
        }
    
    def update_possible_combinations(self, guess, total_correct, positions_correct):
        """更新可能数字组合（全向量化过滤）"""
        if len(self.possible_combinations) == 0:
            return

        guess_arr = np.frombuffer(guess.encode('ascii'), dtype=np.uint8) - 48

        # 先用“位置正确数”做早期裁剪，减少后续工作量
        correct_positions = (self.possible_combinations == guess_arr).sum(axis=1)
        pos_mask = (correct_positions == positions_correct)
        if not np.any(pos_mask):
            # 无匹配，直接置空
            self.possible_combinations = self.possible_combinations[:0]
            return

        cand = self.possible_combinations[pos_mask]
        N, L = cand.shape

        # 为每行构建 0..9 的计数：使用单次 np.bincount 完成
        rows = np.repeat(np.arange(N), L)
        vals = cand.reshape(-1)
        idx = rows * 10 + vals
        counts_flat = np.bincount(idx, minlength=N * 10)
        secret_counts = counts_flat.reshape(N, 10)

        guess_counts = np.bincount(guess_arr, minlength=10)

        total_digits = np.minimum(secret_counts, guess_counts).sum(axis=1)

        mask = (total_digits == total_correct)

        # 组合位置掩码与数字掩码
        final = np.where(pos_mask)[0][mask]
        self.possible_combinations = self.possible_combinations[final]
    
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