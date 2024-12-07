import pickle
import numpy as np
import random

class Agent(object):

    def __init__(self):
        self.table: dir = {} # v值表，从状态映射到v值
        self.epsilon = 0.5
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.count = 5000

    def random_move(self, state, current_player):
        empty_positions = np.argwhere(state == 0)
        # 要有空位
        assert (len(empty_positions)>0)
        move = random.choice(empty_positions)
        move = tuple(move)
        return move

    def policy_move(self, state, current_player, size):
        empty_positions = np.argwhere(state == 0)
        q_values = []
        for empty_position in empty_positions:
            move = empty_position
            move = tuple(move)
            key = state.copy()
            key[move] = current_player
            key = tuple(key.reshape(size * size))
            if key in self.table: # 在v值表中
                black_reward, white_reward = self.table[key]
                # v值采取以下式子：加上自己的奖励，减去对手的奖励
                q_value = (black_reward * current_player - white_reward * current_player)/2
                q_values.append((move, q_value))
        print(q_values) # 训练时可以注释掉

        if not q_values:
            return self.random_move(state , current_player)

        move = max(q_values, key=lambda a: a[1])[0]
        return move

class Game(object):

    def __init__(self):
        self.size = 3 # 棋盘
        self.n_in_a_row = 3 # 胜利条件：三连珠
        self.state = np.zeros((self.size, self.size), dtype= int)
        self.current_player = 1 # black
        self.agent = Agent()
        self.episode = []
        self.alpha = 0.1 # 更新率

    def display(self):
        """
        打印棋盘，黑子显示为 'X'，白子显示为 'O'，空位显示为 '.'
        """
        symbol_map = {1: 'X', -1: 'O', 0: '.'}

        # 打印列号
        print("   ", end="")
        for col in range (self.size):
            print(f"{col} ", end="")
        print()  # 换行
        # 打印每一行及其行号
        for row in range(self.size):
            print(f"{row}  ", end="")  # 打印行号
            for col in range(self.size):
                print(f"{symbol_map[self.state[row][col]]} ", end="")  # 打印棋盘内容
            print()  # 换行
        print()  # 打印空行，分隔每次棋盘的状态

    def input(self):
        # 和人交互，input as the form of: 1 1 , 0 0
        x, y = input("enter the position: ").split()
        action = (int(x) , int(y))
        while self.state[action] != 0:
            print("invalid position")
            x, y = input("enter the position: ").split()
            action = (int(x), int(y))
        return action

    def has_n_in_a_row(self, x, y, n):
        """
        判断 (x, y) 位置的棋子是否形成 n 连珠
        """
        player = self.state[x, y]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 水平，垂直，对角线，反对角线

        for dx, dy in directions:
            count = 1  # 当前棋子的数量
            # 正向查找
            for step in range(1, n):
                nx, ny = x + step * dx, y + step * dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self.state[nx, ny] == player:
                    count += 1
                else:
                    break

            # 反向查找
            for step in range(1, n):
                nx, ny = x - step * dx, y - step * dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self.state[nx, ny] == player:
                    count += 1
                else:
                    break

            # 判断是否达成 n 连珠
            if count >= n:
                return True

        return False

    def winner_reward(self):
        # 检查是否存在n连珠的赢家
        winner = 0
        for x in range(self.size):
            for y in range(self.size):
                if self.state[x, y] != 0 and self.has_n_in_a_row(x, y, self.n_in_a_row):
                    winner = self.state[x, y]  # 赢家（1或-1）

        if winner == 1:
            return np.array([1, -1])
        elif winner == -1:
            return np.array([-1, 1])
        elif len(np.argwhere(self.state == 0)) == 0: # 下满且平局
            return np.array([0, 0])
        else:                                        # 棋局未结束
            return None

    def if_done(self):
        flag = self.winner_reward()
        if flag is None:
            return False
        black, white = flag
        if black == 1:
            print("black win")
        elif white == 1:
            print("white win")
        elif not black and not white:
            print("draw")
        return True

    def make_move(self , action):
        self.state[action] = self.current_player
        self.current_player = -1 * self.current_player

    def selfplay(self):
        reward = None
        state = tuple(self.state.reshape(self.size * self.size))
        self.episode.append(state)
        r = random.random()
        while reward is None:
            if r < self.agent.epsilon:
                move = self.agent.random_move(self.state, self.current_player)
            else:
                move = self.agent.policy_move(self.state, self.current_player, self.size)
            self.make_move(move)
            state = tuple(self.state.reshape(self.size * self.size))
            self.episode.append(state)
            reward = self.winner_reward()
        for key in self.episode:
            self.agent.table.setdefault(key, np.array([0, 0]))
            self.agent.table[key] = self.agent.table[key] - self.alpha * (self.agent.table[key] - reward)

    def train(self):
        for i in range (self.agent.count):
            if i % 1000 == 0:
                print("count: " , i)
            self.state = np.zeros((self.size, self.size), dtype=int)
            self.current_player = 1
            self.episode = []
            self.selfplay()
            # self.agent.epsilon = self.agent.epsilon * self.agent.epsilon_decay
            # if self.agent.epsilon < self.agent.epsilon_min:
            #    self.agent.epsilon = self.agent.epsilon_min


    def human_ai_play(self):
        self.state = np.zeros((self.size, self.size), dtype= int)
        self.current_player = 1
        choice = input("请选择执黑棋(b)还是白棋(w): ")
        if choice == 'b':
            human_player = 1  # 玩家为黑
        else:
            human_player = -1  # 玩家为白

        self.display()
        while True:
            if human_player == 1:
                action = self.input()
                self.make_move(action)
                self.display()
                if self.if_done():
                    break
                action = self.agent.policy_move(self.state, self.current_player, self.size)
                self.make_move(tuple(action))
                self.display()
                if self.if_done():
                    break

            if human_player == -1:
                action = self.agent.policy_move(self.state, self.current_player, self.size)
                self.make_move(tuple(action))
                self.display()
                if self.if_done():
                    break
                action = self.input()
                self.make_move(action)
                self.display()
                if self.if_done():
                    break

    def save_table(self, path):
        with open(path, 'wb') as file:
            file.write(pickle.dumps(self.agent.table))

    def load_table(self, path):
        with open(path, 'rb') as file:
            self.agent.table = pickle.load(file)
