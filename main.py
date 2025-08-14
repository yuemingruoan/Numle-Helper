import numpy as np
import math
import time
import sys

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
                
if __name__ == "__main__":
    print("Numle 助手 - 选择模式:")
    print("1. 检查模式 (check)")
    print("2. 求解模式 (solve)")
    print("3. 自动遍历测试")
    mode = input("请输入模式编号: ").strip()
    
    if mode == "1":
        secret = input("请输入谜底数字: ").strip()
        print("\n进入检查模式，输入'q'退出")
        while True:
            guess = input("请输入猜测数字: ").strip()
            if guess.lower() == 'q':
                print("退出检查模式")
                break
                
            try:
                result = InteractiveNumleSolver.check(secret, guess)
                print(f"结果: 包含数字: {result['total_digits']}, 位置正确: {result['correct_positions']}")
            except ValueError as e:
                print(f"错误: {e}")
    elif mode == "2":
        length = int(input("请输入数字长度: "))
        solver = InteractiveNumleSolver(length)
        solver.solve_interactive()
    elif mode == "3":
        length = int(input("请输入数字长度: "))
        solver = InteractiveNumleSolver(length)
        print("\n开始自动遍历测试...")
        
        start_time = time.time()
        total_tests = len(solver.all_combinations)
        success_count = 0
        total_attempts = 0
        max_attempts = 0
        
        current_test = 0
        try:
            for secret in solver.all_combinations:
                current_test += 1
                print(f"进度: {current_test}/{total_tests}", end="\r")
                solver.possible_combinations = solver.all_combinations.copy()
                attempt = 1
                solved = False
                
                while True:
                    guess = solver.get_next_guess()
                    if guess is None:
                        break
                        
                    result = solver.check(''.join(map(str, secret)), guess)
                    total_correct = result['total_digits']
                    positions_correct = result['correct_positions']
                    
                    if positions_correct == solver.digit_length:
                        solved = True
                        success_count += 1
                        total_attempts += attempt
                        if attempt > max_attempts:
                            max_attempts = attempt
                        break
                        
                    solver.update_possible_combinations(guess, total_correct, positions_correct)
                    attempt += 1
        except KeyboardInterrupt:
            print("\n\n测试被中断，显示当前统计结果:")
            end_time = time.time()
            total_time = end_time - start_time
            print(f"已完成测试数: {current_test}/{total_tests}")
            print(f"成功次数: {success_count}")
            if current_test > 0:
                print(f"当前成功率: {success_count/current_test*100:.2f}%")
            if success_count > 0:
                print(f"平均猜测次数: {total_attempts/success_count:.2f}")
                print(f"最高猜测次数: {max_attempts}")
            print(f"当前总耗时: {total_time:.2f}秒")
            if current_test > 0:
                print(f"平均每个测试耗时: {total_time/current_test:.4f}秒")
            sys.exit(0)
                
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n测试完成！结果:")
        print(f"测试总数: {total_tests}")
        print(f"成功次数: {success_count}")
        print(f"成功率: {success_count/total_tests*100:.2f}%")
        if success_count > 0:
            print(f"平均猜测次数: {total_attempts/success_count:.2f}")
            print(f"最高猜测次数: {max_attempts}")
        print(f"总耗时: {total_time:.2f}秒")
        if total_tests > 0:
            print(f"平均每个测试耗时: {total_time/total_tests:.4f}秒")
    else:
        print("无效模式选择")