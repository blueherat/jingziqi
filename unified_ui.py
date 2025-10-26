import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import numpy as np
import threading
import time
import os
import torch
import pickle
import joblib

# --- 模型文件路径配置 ---
# 在这里直接指定您的模型文件路径。
# 程序将根据这些路径自动加载模型。
MODEL_PATHS = {
    "Deep TD": "deeptd_test1.pkl",  # 您的 DeepTD 模型文件
    "Monte Carlo": "3_3_3_final_mc.pkl",  # 您的 Monte Carlo 模型文件
    "Tabular TD": "3_3_td_50w.pkl"  # 您的 Tabular TD 模型文件
}
# -------------------------

# 尝试导入您的三个算法文件
try:
    import deepTD

    DEEPTD_AVAILABLE = True
except ImportError:
    DEEPTD_AVAILABLE = False
    print("Warning: deepTD.py not found.")

try:
    import mc

    MC_AVAILABLE = True
except ImportError:
    MC_AVAILABLE = False
    print("Warning: mc.py not found.")

try:
    import TD

    TD_AVAILABLE = True
except ImportError:
    TD_AVAILABLE = False
    print("Warning: TD.py not found.")


# ==============================================================================
# 1. 通用游戏核心 (Game Core) - [此部分代码与之前版本相同，保持不变]
# ==============================================================================
class TicTacToeCore:
    def __init__(self, size=3):
        self.size = size
        self.reset()

    def reset(self):
        self.state = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def make_move(self, row, col):
        if self.state[row, col] == 0 and not self.game_over:
            self.state[row, col] = self.current_player
            self.check_winner()
            self.current_player *= -1
            return True
        return False

    def check_winner(self):
        for i in range(self.size):
            if abs(sum(self.state[i, :])) == self.size:
                self.winner = self.state[i, 0];
                self.game_over = True;
                return
            if abs(sum(self.state[:, i])) == self.size:
                self.winner = self.state[0, i];
                self.game_over = True;
                return
        diag1 = sum([self.state[i, i] for i in range(self.size)])
        diag2 = sum([self.state[i, self.size - 1 - i] for i in range(self.size)])
        if abs(diag1) == self.size:
            self.winner = self.state[0, 0];
            self.game_over = True;
            return
        if abs(diag2) == self.size:
            self.winner = self.state[0, self.size - 1];
            self.game_over = True;
            return
        if len(np.argwhere(self.state == 0)) == 0:
            self.winner = 0;
            self.game_over = True;
            return


# ==============================================================================
# 2. AI 适配器 (AI Adapters) - [此部分代码与之前版本相同，保持不变]
# ==============================================================================
class AIAdapter:
    def load_model(self, path): raise NotImplementedError

    def get_move(self, state, current_player): raise NotImplementedError


class DeepTDAdapter(AIAdapter):
    def __init__(self, size=3):
        if not DEEPTD_AVAILABLE: raise ImportError("deepTD.py is missing")
        self.agent = deepTD.Agent(size)
        self.size = size

    def load_model(self, path):
        self.agent.vnetwork.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.agent.vnetwork.eval()

    def get_move(self, state, current_player):
        return self.agent.policy_move(state, current_player)  # DeepTD的policy_move不需要size参数


class MCAdapter(AIAdapter):
    def __init__(self, size=3):
        if not MC_AVAILABLE: raise ImportError("mc.py is missing")
        self.agent = mc.Agent()
        self.size = size

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.agent.table = pickle.load(f)

    def get_move(self, state, current_player):
        return self.agent.policy_move(state, current_player, self.size)


class TDAdapter(AIAdapter):
    def __init__(self, size=3):
        if not TD_AVAILABLE: raise ImportError("TD.py is missing")
        self.agent = TD.Agent()
        self.size = size

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.agent.table = joblib.load(f)

    def get_move(self, state, current_player):
        return self.agent.policy_move(state, current_player, self.size)


# ==============================================================================
# 3. 统一图形界面 (Unified GUI) - [此部分代码已修改]
# ==============================================================================
class UnifiedGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("强化学习井字棋 - 统一测试平台")
        self.root.resizable(False, False)

        self.game = TicTacToeCore(size=3)
        self.current_ai = None
        self.human_side = 1

        self.setup_ui()
        # *** 修改点: 程序启动时自动加载默认算法的模型 ***
        self.on_algo_change()

    def setup_ui(self):
        # --- 顶部控制区 ---
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(fill=tk.X)

        # 算法选择
        tk.Label(control_frame, text="选择算法:").grid(row=0, column=0, sticky=tk.W)
        self.algo_var = tk.StringVar()
        self.algo_combo = ttk.Combobox(control_frame, textvariable=self.algo_var, state="readonly")

        # *** 修改点: 动态填充下拉菜单的值 ***
        available_algos = []
        if DEEPTD_AVAILABLE: available_algos.append('Deep TD')
        if MC_AVAILABLE: available_algos.append('Monte Carlo')
        if TD_AVAILABLE: available_algos.append('Tabular TD')
        self.algo_combo['values'] = available_algos

        if available_algos:
            self.algo_combo.current(0)
        self.algo_combo.grid(row=0, column=1, padx=5, sticky=tk.W, columnspan=2)
        self.algo_combo.bind("<<ComboboxSelected>>", self.on_algo_change)

        # 模型状态标签
        self.model_status_var = tk.StringVar(value="正在初始化...")
        self.model_status_label = tk.Label(control_frame, textvariable=self.model_status_var, fg="blue")
        self.model_status_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))

        # --- 中间棋盘区 ---
        board_frame = tk.Frame(self.root, padx=20, pady=10)
        board_frame.pack()

        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        for r in range(3):
            for c in range(3):
                btn = tk.Button(board_frame, text="", font=('Arial', 36, 'bold'), width=4, height=2,
                                command=lambda row=r, col=c: self.on_click(row, col))
                btn.grid(row=r, column=c, padx=2, pady=2)
                self.buttons[r][c] = btn

        # --- 底部状态与操作区 ---
        bottom_frame = tk.Frame(self.root, padx=10, pady=10)
        bottom_frame.pack(fill=tk.X)

        self.game_status_var = tk.StringVar(value="请选择算法开始游戏")
        tk.Label(bottom_frame, textvariable=self.game_status_var, font=('Arial', 12)).pack(pady=5)

        btn_frame = tk.Frame(bottom_frame)
        btn_frame.pack()
        tk.Button(btn_frame, text="重新开始", command=self.reset_game, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="切换执方", command=self.switch_sides, width=12).pack(side=tk.LEFT, padx=5)

    def on_algo_change(self, event=None):
        """当算法选择改变时，自动加载对应的预设模型"""
        algo_key = self.algo_var.get()
        if not algo_key:
            messagebox.showwarning("无可用算法", "没有找到任何可用的算法文件 (deepTD.py, mc.py, TD.py)。")
            return

        model_path = MODEL_PATHS.get(algo_key)

        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("模型文件错误", f"找不到为 '{algo_key}' 配置的模型文件:\n'{model_path}'\n请检查 MODEL_PATHS 配置和文件是否存在。")
            self.current_ai = None
            self.model_status_var.set(f"模型文件 '{model_path}' 不存在")
            self.model_status_label.config(fg="red")
            self.reset_game()
            return

        try:
            if "Deep TD" in algo_key:
                self.current_ai = DeepTDAdapter()
            elif "Monte Carlo" in algo_key:
                self.current_ai = MCAdapter()
            elif "Tabular TD" in algo_key:
                self.current_ai = TDAdapter()

            self.current_ai.load_model(model_path)
            self.model_status_var.set(f"算法: {algo_key} | 模型: {os.path.basename(model_path)}")
            self.model_status_label.config(fg="green")
            self.reset_game()

        except ImportError as e:
            messagebox.showerror("错误", f"缺少必要的文件: {e}\n请确保 deepTD.py, mc.py, TD.py 在同一目录下。")
        except Exception as e:
            messagebox.showerror("加载失败", f"加载模型 '{model_path}' 失败:\n{e}")
            self.current_ai = None
            self.model_status_var.set("加载失败")
            self.model_status_label.config(fg="red")
            self.reset_game()

    # [ reset_game, switch_sides, set_board_enabled, on_click, ai_move, check_game_over, update_board_ui, highlight_winner ]
    # 这些方法与之前版本相同，无需修改，此处为简洁省略。
    # 请从上一版代码中复制这些方法到此处。
    def reset_game(self):
        self.game.reset()
        self.update_board_ui()
        for r in range(3):
            for c in range(3):
                self.buttons[r][c].config(state=tk.NORMAL, bg="#f0f0f0")

        if not self.current_ai:
            self.game_status_var.set("模型加载失败，请检查配置")
            self.set_board_enabled(False)
            return

        if self.game.current_player == self.human_side:
            self.game_status_var.set("轮到你了 (执 %s)" % ("黑[X]" if self.human_side == 1 else "白[O]"))
            self.set_board_enabled(True)
        else:
            self.game_status_var.set("AI 思考中...")
            self.set_board_enabled(False)
            self.root.after(500, self.ai_move)

    def switch_sides(self):
        self.human_side *= -1
        self.reset_game()

    def set_board_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        for r in range(3):
            for c in range(3):
                if self.buttons[r][c]['text'] == "":
                    self.buttons[r][c].config(state=state)

    def on_click(self, row, col):
        if self.game.make_move(row, col):
            self.update_board_ui()
            if self.check_game_over():
                return

            self.game_status_var.set("AI 思考中...")
            self.set_board_enabled(False)
            self.root.after(500, self.ai_move)

    def ai_move(self):
        if not self.current_ai or self.game.game_over: return

        try:
            move = self.current_ai.get_move(self.game.state, self.game.current_player)
            self.game.make_move(move[0], move[1])
            self.update_board_ui()
            if self.check_game_over():
                return

            self.game_status_var.set("轮到你了")
            self.set_board_enabled(True)
        except Exception as e:
            messagebox.showerror("AI错误", f"AI决策时发生错误: {e}")
            self.reset_game()

    def check_game_over(self):
        if self.game.game_over:
            if self.game.winner == 0:
                msg = "平局！"
            elif self.game.winner == self.human_side:
                msg = "恭喜！你赢了！"
            else:
                msg = "遗憾，AI赢了。"
            self.game_status_var.set(msg)
            self.set_board_enabled(False)
            messagebox.showinfo("游戏结束", msg)
            return True
        return False

    def update_board_ui(self):
        sym_map = {1: 'X', -1: 'O', 0: ''}
        color_map = {1: 'black', -1: 'red', 0: 'black'}
        for r in range(3):
            for c in range(3):
                val = self.game.state[r, c]
                self.buttons[r][c].config(text=sym_map[val], disabledforeground=color_map[val])


if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedGUI(root)
    # 居中显示
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    root.mainloop()