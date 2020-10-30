from envs.grid_game_flat import GridGame
from utils.dotdic import DotDic
import json
import torch

opt = DotDic(json.loads(open('config/grid_3_dial.json', 'r').read()))
opt.bs = 3
opt.game_nagents = 4
opt.game_action_space_total = 6


g = GridGame(opt, (4, 4))
g.show(vid=False)
u = torch.zeros((opt.bs, opt.game_nagents)) + 4
g.get_reward(u)
g.show(vid=False)
# print(g.get_action_range(None, None))
