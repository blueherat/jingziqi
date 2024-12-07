"""
1.使用任一模型
2.注释对应文件（1）policy_move的print，否则会打印每步value；
            （2）if_done中的print，否则会显示胜利方
3.如果是table，可以打印table个数
"""
import os
from deepTD import Game
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(dir_path, "deeptd_test1.pkl")
game = Game()
game.train()
# print(len(game.agent.table)) # 打印table中的状态数
plt.figure()
plt.plot(game.loss)
plt.show()
game.save_model(save_path)