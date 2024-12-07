import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class Vnetwork(nn.Module):
    def __init__(self, input_size, output_size= 2):
        super(Vnetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) # 不能加relu，否则将会使得最后一层永远输出不了负数
        return x

class Agent(object):

    def __init__(self, size, learning_rate= 0.001):
        self.size = size
        # self.table: dir = {} # v值表，从状态映射到v值
        self.vnetwork = Vnetwork(self.size * self.size, 2)
        self.target_vnetwork = Vnetwork(self.size * self.size, 2)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.vnetwork.parameters(), lr=learning_rate)
        self.epsilon = 0.5
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.count = 50000

    def random_move(self, state, current_player):
        empty_positions = np.argwhere(state == 0)
        # 要有空位
        assert (len(empty_positions)>0)
        move = random.choice(empty_positions)
        move = tuple(move)
        return move

    def policy_move(self, state, current_player, size = 3):
        empty_positions = np.argwhere(state == 0)
        q_values = []
        for empty_position in empty_positions:
            move = empty_position
            move = tuple(move)
            key = state.copy()
            key[move] = current_player
            key = torch.FloatTensor(tuple(key.reshape(self.size * self.size)))
            black_reward, white_reward = self.vnetwork(key)
            # v值采取以下式子：加上自己的奖励，减去对手的奖励
            if current_player == 1:
                q_value = black_reward
            else:
                q_value = white_reward
            q_values.append((move, q_value))
        # print(q_values) # 训练时可以注释掉

        move = max(q_values, key=lambda a: a[1])[0]
        return move

class Game(object):

    def __init__(self):
        self.size = 3 # 棋盘
        self.n_in_a_row = 3 # 胜利条件：三连珠
        self.state = np.zeros((self.size, self.size), dtype= int)
        self.pre_state = np.zeros((self.size, self.size), dtype= int)
        # self.present_reward = np.array([0, 0, 0])
        self.reward = np.array([0, 0, 0])
        self.current_player = 1 # black
        self.agent = Agent(self.size)
        self.episode = []
        self.alpha = 0.1 # 更新率
        self.gamma = 0.96
        self.batch_size = 64
        self.memorys = deque(maxlen=10000000)
        self.loss= []

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
            return np.array([1, -1, 0])
        elif winner == -1:
            return np.array([-1, 1, 0])
        elif len(np.argwhere(self.state == 0)) == 0: # 下满且平局
            return np.array([0, 0, 1])
        else:                                        # 棋局未结束
            return np.array([0, 0, 0])

    def if_done(self):
        flag = self.winner_reward()
        black, white, draw = flag
        if black == 1:
            # print("black win")
            return True
        elif white == 1:
            # print("white win")
            return True
        elif draw == 1:
            # print("draw")
            return True
        else:
            return False

    def make_move(self , action):
        self.state[action] = self.current_player
        self.current_player = -1 * self.current_player

    def selfplay(self):
        done = False
        self.pre_state = None
        self.reward = self.winner_reward()
        # state = tuple(self.state.reshape(9))
        # 开始时状态
        state_pair = (tuple(self.state.reshape(self.size * self.size)), None, self.reward)
        self.episode.append(state_pair)
        self.pre_state = self.state
        while not done:
            # self.display()
            self.pre_state = self.state.copy()
            if random.random() < self.agent.epsilon:
                move = self.agent.random_move(self.state, self.current_player)
            else:
                move = self.agent.policy_move(self.state, self.current_player, self.size)
            # print(move)
            self.make_move(move)
            self.reward = self.winner_reward()
            done = self.if_done()
            # state = tuple(self.state.reshape(9))
            state_pair = (
                tuple(self.state.reshape(self.size * self.size)), tuple(self.pre_state.reshape(self.size * self.size)),
                self.reward)
            self.episode.append(state_pair)

        # print(self.episode)
        # states = []
        # td_targets = []
        memory_k = []
        self.episode = self.episode[::-1]
        for one in self.episode:
            state, pre_state, reward = one
            black, white, draw = reward

            if black or white or draw:
                done = True
            else:
                done = False
            if done:
                # td_target = (black, white)
                # current_value = self.agent.vnetwork(state)
                # print("td_target_0", td_target)
                # print("current_value_0", current_value)
                # states.append(state)
                # td_targets.append(td_target)
                reward = (black, white)
                memory = (state, state, done, reward)
                # print(memory)
                self.memorys.append(memory)
                memory_k.append(memory)
                memory = (pre_state, state, not done, (0, 0))
                self.memorys.append(memory)
                memory_k.append(memory)
                # print(memory)
            if pre_state is not None and not done:
                state_tenor = torch.FloatTensor(state)
                td_target = 0 + self.gamma * self.agent.vnetwork(state_tenor)
                # td_target = td_target.detach.numpy()
                # current_value = self.agent.vnetwork(pre_state)
                # print("td_target", td_target)
                # print("current_value", current_value)
                # states.append(pre_state)
                # td_targets.append(td_target)
                reward = (0, 0)
                memory = (pre_state, state, done, reward)
                # print(memory)
                self.memorys.append(memory)
                memory_k.append(memory)
        """
        if len(self.memorys) < self.batch_size:
            return
        batch = random.sample(self.memorys, self.batch_size)
        """
        pre_states ,states, dones, rewards = zip(*memory_k)
        pre_states_tensor = torch.FloatTensor(pre_states)
        states_tensor = torch.FloatTensor(states)
        rewards_tensor = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        dones_tensor = dones.view(-1, 1)

        td_targets_tensor =self.gamma * self.agent.target_vnetwork(states_tensor) * (1-dones_tensor) + 1 * rewards_tensor
        # print(td_targets_tensor.min(), td_targets_tensor.max())

        current_values_tensor = self.agent.vnetwork(pre_states_tensor)

        # torch.nn.utils.clip_grad_norm_(self.agent.vnetwork.parameters(), max_norm=1.0)
        loss = nn.MSELoss()(current_values_tensor, td_targets_tensor)
        self.agent.optimizer.zero_grad()
        loss.backward()
        self.agent.optimizer.step()
        return loss.item()

    def train(self):
        for i in range (self.agent.count):
            if i % 200 == 0 or i < 200:
                print("count: " , i)
                self.state = np.zeros((self.size, self.size), dtype=int)
                self.current_player = 1
                self.episode = []
                loss = self.selfplay()
                print(loss)
                self.loss.append(loss)
                self.agent.target_vnetwork.load_state_dict(self.agent.vnetwork.state_dict())
            else:
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
                action = self.agent.policy_move(self.state, self.current_player,self.size)
                self.make_move(tuple(action))
                self.display()
                if self.if_done():
                    break
                action = self.input()
                self.make_move(action)
                self.display()
                if self.if_done():
                    break

    def save_model(self, path):
        with open(path, 'wb') as file:
            torch.save(self.agent.vnetwork.state_dict(), file)
            print("save successfully")

    def load_model(self, path):
        with open(path, 'rb') as file:
            self.agent.vnetwork.load_state_dict(torch.load(file))
