from envs.grid_game import GridGame
from utils.dotdic import DotDic
import json
import torch

opt = DotDic(json.loads(open('config/switch_3_dial.json', 'r').read()))
opt.bs = 3
opt.game_nagents = 4

g = GridGame(opt, (4, 4))
# g.show(vid=False)
# u = torch.zeros((1, 4)) + 0
# g.get_reward(u)
# g.show(vid=False)
print(g.get_action_range())
