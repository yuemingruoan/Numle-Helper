from collections import defaultdict

class InteractiveWordleSolver:
    def __init__(self, word_length=5):
        self.word_length = word_length
        self.filename = f"split_results/words_len_{word_length}.txt"
        self.possible_words = []  # 可能单词集合（动态加载）
        
    def load_word_list(self):
        """从文件加载单词列表"""
        with open(self.filename, 'r') as f:
            return [line.strip() for line in f if len(line.strip()) == self.word_length]
    
    def get_next_guess(self):
        """获取下一个猜测（基于字母频率）"""
        # 如果可能单词集为空，重新加载文件
        if not self.possible_words:
            self.possible_words = self.load_word_list()
            
        if not self.possible_words:
            return None
            
        # 计算字母位置频率
        freq = [defaultdict(int) for _ in range(self.word_length)]
        for word in self.possible_words:
            for i, char in enumerate(word):
                freq[i][char] += 1
                
        # 选择总分最高的单词
        best_word = None
        best_score = -1
        
        for word in self.possible_words:
            score = sum(freq[i][char] for i, char in enumerate(word))
            if score > best_score:
                best_score = score
                best_word = word
                
        return best_word
    
    def update_possible_words(self, guess, feedback):
        """根据反馈更新可能单词集合"""
        # 重新从文件加载所有单词
        all_words = self.load_word_list()
        new_possible = []
        
        for word in all_words:
            # 跳过已被排除的单词
            if word in self.possible_words or not self.possible_words:
                valid = True
                # 临时计数器（用于处理重复字母）
                word_counter = defaultdict(int)
                for char in word:
                    word_counter[char] += 1
                    
                # 检查绿色（正确位置）
                for i, (g_char, fb) in enumerate(zip(guess, feedback)):
                    if fb == 'G':
                        if word[i] != g_char:
                            valid = False
                            break
                        else:
                            word_counter[g_char] -= 1
                
                if not valid:
                    continue
                    
                # 检查黄色（存在但位置错误）
                for i, (g_char, fb) in enumerate(zip(guess, feedback)):
                    if fb == 'Y':
                        if word[i] == g_char or g_char not in word or word_counter[g_char] == 0:
                            valid = False
                            break
                        else:
                            word_counter[g_char] -= 1
                
                if not valid:
                    continue
                    
                # 检查灰色（不存在）
                for i, (g_char, fb) in enumerate(zip(guess, feedback)):
                    if fb == 'X':
                        if g_char in word and word_counter[g_char] > 0:
                            valid = False
                            break
                
                if valid:
                    new_possible.append(word)
                    
        self.possible_words = new_possible
    
    def solve_interactive(self, max_attempts=6):
        """交互式求解主函数"""
        print(f"Wordle 求解器已启动（{self.word_length}字母单词）")
        print(f"使用单词表: {self.filename}")
        
        attempt = 1
        while attempt <= max_attempts:
            # 初始加载单词表
            if not self.possible_words:
                self.possible_words = self.load_word_list()
                
            guess = self.get_next_guess()
            if guess is None:
                print("无有效单词可猜，游戏结束")
                return False
                
            print(f"\n尝试 #{attempt}: 我猜: {guess.upper()}")
            
            while True:
                feedback = input("请输入反馈（G=绿/Y=黄/X=灰），或输入'no'表示单词不存在: ").strip().upper()
                
                if feedback == 'NO':
                    print(f"已标记 {guess} 为无效单词")
                    # 直接从当前可能单词集中移除
                    if guess in self.possible_words:
                        self.possible_words.remove(guess)
                    break
                
                # 验证反馈格式
                if len(feedback) != self.word_length or any(char not in 'GYX' for char in feedback):
                    print(f"格式错误！请输入{self.word_length}位G/Y/X组合（例如：GXXYG）")
                    continue
                
                # 处理有效反馈
                if feedback == 'G' * self.word_length:
                    print(f"\n成功！单词是：{guess.upper()}")
                    return True
                    
                self.update_possible_words(guess, feedback)
                print(f"剩余可能单词数: {len(self.possible_words)}")
                attempt += 1  # 消耗尝试次数
                break
                
        print("\n未能在指定次数内猜出单词")
        return False

if __name__ == "__main__":
    long=int(input("请输入单词长度："))
    solver = InteractiveWordleSolver(long)
    solver.solve_interactive()