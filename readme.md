# 强化学习演示平台

## 项目简介
这是一个带有 Tkinter UI 的强化学习演示平台，旨在展示不同强化学习算法在井字棋游戏中的应用。用户可以训练 AI、选择不同的算法与 AI 对战，观察 AI 的决策过程和游戏结果。

## 算法说明
本项目实现了以下三种强化学习算法：

1. **Monte Carlo（蒙特卡罗方法）**: 通过模拟多次游戏来评估每个状态的价值，采用 $\epsilon$-贪心策略进行决策。根据完整回合的最终奖励更新状态价值。

2. **Tabular TD（表格时间差分）**: 使用表格存储状态值，基于单步时间差分方法进行学习和更新。通过比较当前状态值和下一状态值的差异来更新价值估计。

3. **Deep TD（深度时间差分）**: 基于深度学习的时间差分学习算法，使用神经网络来估计状态值。通过构建多层感知器网络和目标网络实现稳定的学习过程。

## 核心代码文件说明

### [mc.py](mc.py)
Monte Carlo 算法实现：
- **Agent 类**: 维护一个字典表来存储每个棋盘状态的黑白两色价值评估。通过 `policy_move()` 方法评估所有可行动作后的状态价值，选择最优动作。采用 $\epsilon$-贪心策略平衡探索与利用。
- **Game 类**: 实现游戏逻辑和自对弈训练。在每次完整回合结束后，根据游戏的最终结果（黑胜、白胜或平局）对回合中访问过的所有状态进行价值更新。

### [TD.py](TD.py)
Tabular TD 算法实现：
- **Agent 类**: 使用字典表存储每个棋盘状态对应的黑白两色价值。通过 `policy_move()` 方法在落子后评估新状态的价值，选择价值最高的动作。
- **Game 类**: 实现游戏逻辑和自对弈训练。采用单步时间差分更新方式，在每次落子后立即根据当前状态值和下一状态值来更新前一步的价值估计。相比 Monte Carlo，TD 方法可以在游戏进行中逐步学习。

### [deepTD.py](deepTD.py)
Deep TD 算法实现：
- **Vnetwork 类**: 定义了一个多层感知器神经网络模型。网络接收 9 维的棋盘状态向量作为输入，通过两层隐藏层处理，输出 2 维向量分别表示黑色和白色的价值评估。
- **Agent 类**: 管理两个相同结构的网络（主网络和目标网络）以及 Adam 优化器。主网络用于生成动作决策，目标网络用于生成稳定的价值目标，定期同步两个网络的权重以提升训练稳定性。通过 `policy_move()` 方法利用神经网络评估所有可行动作的价值。
- **Game 类**: 实现游戏逻辑和神经网络训练。将每一步的游戏经验（状态、后继状态、奖励、是否结束）存储到经验回放池中。通过批量采样历史经验计算损失函数，使用梯度下降更新网络参数。相比表格方法，神经网络能够处理更大的状态空间并进行泛化。

### [ui.py](ui.py)
统一的 Tkinter 图形界面，提供：
- **TicTacToeCore 类**: 管理游戏的基本逻辑，包括棋盘状态、落子、胜负判定。
- **AIAdapter 及其子类**: 为三种算法提供统一的加载与决策接口（DeepTDAdapter、MCAdapter、TDAdapter）。
- **UnifiedGUI 类**: 提供图形界面，包括算法选择、棋盘交互、游戏状态显示、自动加载模型文件等功能。

### [train.py](train.py)
模型训练脚本，用于启动训练流程。支持修改导入语句来训练不同的算法模型。

### [human_ai_play.py](human_ai_play.py)
命令行版人机对战脚本，支持在终端中与训练好的 AI 进行对战。

## 训练指南

### 准备工作

在训练之前，请确保已安装所需的依赖包：

```bash
pip install numpy torch matplotlib joblib
```

### 训练步骤

#### 1. 修改 [train.py](train.py) 选择训练算法

打开 [train.py](train.py)，在文件开头修改导入语句和保存路径来选择要训练的算法。

**Deep TD 训练**（默认配置）：
```python
from deepTD import Game
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(dir_path, "deeptd_test1.pkl")

game = Game()
game.train()
plt.figure()
plt.plot(game.loss)
plt.show()
game.save_model(save_path)
```

**Monte Carlo 训练**：
```python
from mc import Game

dir_path = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(dir_path, "mc_model.pkl")

game = Game()
game.train()
game.save_table(save_path)
```

**Tabular TD 训练**：
```python
from TD import Game

dir_path = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(dir_path, "td_model.pkl")

game = Game()
game.train()
game.save_table(save_path)
```

#### 2. 调整训练超参数

在 [deepTD.py](deepTD.py)、[TD.py](TD.py) 或 [mc.py](mc.py) 中的 `Agent` 类可修改以下超参数：

```python
self.epsilon = 0.5           # 探索率（初值）
self.epsilon_decay = 0.9999  # 探索率衰减系数
self.epsilon_min = 0.01      # 最小探索率
self.count = 50000           # 训练回合数
```

在 `Game` 类中可调整学习参数：

```python
self.alpha = 0.1  # 更新率（学习速度）
self.gamma = 0.96 # 时间折扣因子
```

Deep TD 特有参数：
```python
self.batch_size = 64         # 批处理大小
self.memorys = deque(maxlen=10000000)  # 经验回放池容量
```

#### 3. 运行训练

执行 [train.py](train.py) 开始训练：

```sh
python train.py
```

训练过程中会输出进度信息。训练完成后，模型权重或状态表会自动保存为 `.pkl` 文件。

**Deep TD 训练特点**：
- 会绘制训练过程中的损失曲线
- 每 200 个回合输出一次损失值
- 定期同步目标网络权重，确保训练稳定性

**Monte Carlo 和 Tabular TD 训练特点**：
- 每 1000 个回合输出一次进度
- 在每个回合完成后更新价值表

#### 4. 注释掉调试输出（可选）

为了加速训练，可以在相应的实现文件中注释掉以下调试输出：

在 [deepTD.py](deepTD.py)、[mc.py](mc.py) 或 [TD.py](TD.py) 中：
- 注释 `policy_move()` 方法中的 `print(q_values)`
- 注释 `if_done()` 方法中的 `print("black win")` 等输出

### 训练完成后的配置

#### 更新模型路径

打开 [ui.py](ui.py)，找到 `MODEL_PATHS` 字典，将生成的模型文件名添加进去：

```python
MODEL_PATHS = {
    "Deep TD": "deeptd_test1.pkl",      # 修改为你的 Deep TD 模型文件
    "Monte Carlo": "mc_model.pkl",      # 修改为你的 MC 模型文件
    "Tabular TD": "td_model.pkl"        # 修改为你的 TD 模型文件
}
```

确保文件路径正确，模型文件与 [ui.py](ui.py) 在同一目录中。

## 使用指南

### 启动游戏 UI

完成模型训练并配置好 `MODEL_PATHS` 后，运行以下命令启动图形界面：

```sh
python ui.py
```

### 界面说明

**界面布局**：
- **顶部**：算法选择下拉框和模型加载状态显示
- **中央**：3×3 棋盘，点击空白格子落子
- **下方**：游戏状态提示和控制按钮

### 游戏操作

1. **选择算法**：
   - 在顶部下拉框中选择要对阵的算法（Deep TD、Monte Carlo 或 Tabular TD）
   - 系统会自动根据 `MODEL_PATHS` 配置加载对应的模型文件
   - 若模型文件不存在，界面会显示错误提示

2. **选择执方**：
   - 点击"切换执方"按钮可在执黑棋(X)和执白棋(O)之间切换
   - 黑棋先手
   - 初始化或切换算法时，系统会自动根据选择的执方决定谁先落子

3. **游戏进行**：
   - 如果轮到你，点击棋盘上的空位落子
   - AI 自动思考后进行回应
   - 观察游戏结果（赢/负/平）

4. **重新开始**：
   - 点击"重新开始"按钮重置棋盘并开始新一局
   - 保持当前选择的算法和执方

### 人机对战（命令行版）

运行 [human_ai_play.py](human_ai_play.py) 进行命令行版的人机对战：

```sh
python human_ai_play.py
```

**操作步骤**：
1. 按照提示选择执黑棋(b)还是白棋(w)
2. 按 "行 列" 的格式输入落子位置（例如：`1 1`）
3. AI 会自动回应你的落子

**注意**：
- 命令行版使用的模型文件在 [human_ai_play.py](human_ai_play.py) 的 `load_path` 变量中配置
- 修改导入语句可以切换使用不同的算法（Deep TD、MC 或 Tabular TD）

## 项目结构

```
.
├── deepTD.py              # Deep TD 算法实现
├── mc.py                  # Monte Carlo 算法实现
├── TD.py                  # Tabular TD 算法实现
├── ui.py                  # Tkinter 游戏界面
├── train.py               # 训练脚本
├── human_ai_play.py       # 人机对战脚本（命令行版）
├── readme.md              # 项目说明文档
└── *.pkl                  # 预训练模型文件
```

## 工作流示例

### 从零开始训练并游玩

```bash
# 1. 训练 Deep TD 模型（train.py 默认配置）
python train.py
# 生成 deeptd_test1.pkl 文件

# 2. 验证模型路径在 ui.py 中已配置
# 检查 MODEL_PATHS 字典中的文件名是否正确

# 3. 启动游戏界面
python ui.py
# 选择 "Deep TD" 开始游玩

# 4. (可选) 训练其他算法
# 修改 train.py 的导入语句，然后重新运行
# 再在 ui.py 中配置新模型的路径
```

### 使用预训练模型

项目已提供预训练的模型文件（若存在）。只需确保 `MODEL_PATHS` 中的文件名与项目目录中的文件名匹配，然后直接运行 `python ui.py` 即可。