import pytest
from misc.game_data import GameData
import numpy as np

def test_constructor():
    print("======= Constructor GameData =======")
    data = GameData()
    assert True
    
# features = np.asarray([-1.0, 1.0, 0.0, -1.0, -1.0])
# label = np.asarray([0.0, 1.0, 0.0, 0.0])
# board_state = "1.0,0.0,-1.0,-1.0:0.0,1.0,0.0,0.0"
# print("======= ADD DATA =======")
# # GameData.add_data(features, label)
# print("======= ENCODE =======")
# print(GameData._encode(features, label))
# board_state = GameData._encode(features, label)
# print("======= DECODE =======")
# print(GameData._decode("-1, 0,-1,1,0:1.,0.,0.,0.,0."))
# data = np.array(yourlist, dtype=np.float32)
# GameData.add_data(data)