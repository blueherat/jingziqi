"""
1.找到合适的table/model
2.import合适的文件,load对应的模型
3.注释对应文件（1）policy_move的print，以打印每步value；
            （2）if_done中的print，以显示胜利方
"""
from deepTD import Game
import os
dir_path = os.path.dirname(os.path.abspath(__file__))
load_path = os.path.join(dir_path, "deeptd_test1.pkl")

game = Game()
game.load_model(load_path)
game.human_ai_play()