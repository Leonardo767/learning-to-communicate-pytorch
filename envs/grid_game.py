from utils.dotdic import DotDic
import numpy as np
import torch
import os


class GridGame:
    def __init__(self, opt, size):
        self.opt = opt
        self.game_actions = DotDic({
            'NOTHING': 0,
            'UP': 1,
            'DOWN': 2,
            'LEFT': 3,
            'RIGHT': 4
        })
        if self.opt.game_action_space != len(self.game_actions):
            raise ValueError(
                "Config action space doesn't match game's ({} != {}).".format(
                    self.opt.game_action_space, len(self.game_actions)))

        self.H = size[0]
        self.W = size[1]
        self.goal_reward = 10
        self.grid = [
            [0 for _ in range(self.W)] for _ in range(self.H)]
        self.reset()

    def reset(self):
        # episode flags
        self.step_count = 0
        self.reward = torch.zeros(self.opt.bs)
        self.terminal = torch.zeros(self.opt.bs, dtype=torch.long)
        # init common reward one-hot (H x W)
        self.grid = torch.zeros(self.H, self.W)
        self.rx = np.random.randint(0, self.W)
        self.ry = np.random.randint(0, self.H)
        self.grid[self.ry, self.rx] = self.goal_reward
        # init agent locations (bs x n x H x W)
        bs = self.opt.bs
        n_agents = self.opt.game_nagents
        if n_agents > self.H * self.W:
            raise ValueError("Too many agents to fit inside grid.")
        self.agent_locs = torch.zeros(bs, n_agents, self.H, self.W)
        self.all_agents_map = torch.zeros(bs, self.H, self.W)
        for b in range(bs):
            # per batch, build world of n_agents, shuffle them around the grid
            agent_world = torch.zeros(self.H * self.W)
            agent_world[0:n_agents] = torch.ones(n_agents)
            idx = torch.randperm(agent_world.nelement())
            agent_world = agent_world[idx]
            agent_world = torch.reshape(agent_world, (self.H, self.W))
            self.all_agents_map[b] = agent_world  # shuffled agent map
            agent_locs_yx = torch.nonzero(agent_world, as_tuple=False)
            for n in range(n_agents):
                y_loc, x_loc = agent_locs_yx[n][0], agent_locs_yx[n][1]
                self.agent_locs[b, n, y_loc, x_loc] = 1

    def get_reward(self, a_t):
        # all actions cost -1 reward
        for b in range(self.opt.bs):
            self.reward[b] = self.reward[b] - 1
            for n in range(self.opt.game_nagents):
                # assess current location
                curr_loc = torch.nonzero(
                    self.agent_locs[b, n], as_tuple=False)[0]
                on_left_edge = curr_loc[1].item() == 0
                on_right_edge = curr_loc[1].item() == self.W - 1
                on_top_edge = curr_loc[0].item() == 0
                on_bottom_edge = curr_loc[0].item() == self.H - 1
                proposed_action = int(a_t[b, n].item())
                if self._agents_will_collide(b, curr_loc, proposed_action):
                    continue
                # move agent [b, n] according to action
                old_loc_map = self.agent_locs[b, n]
                self.all_agents_map[b] = self.all_agents_map[b] - old_loc_map
                if proposed_action == self.game_actions.UP and not on_top_edge:
                    self.agent_locs[b, n, :, :] = torch.roll(
                        self.agent_locs[b, n], -1, dims=0)
                elif proposed_action == self.game_actions.DOWN and not on_bottom_edge:
                    self.agent_locs[b, n, :, :] = torch.roll(
                        self.agent_locs[b, n], 1, dims=0)
                elif proposed_action == self.game_actions.LEFT and not on_left_edge:
                    self.agent_locs[b, n, :, :] = torch.roll(
                        self.agent_locs[b, n], -1, dims=1)
                elif proposed_action == self.game_actions.RIGHT and not on_right_edge:
                    self.agent_locs[b, n, :, :] = torch.roll(
                        self.agent_locs[b, n], 1, dims=1)
                # update agents map for the batch
                new_loc_map = self.agent_locs[b, n]
                self.all_agents_map[b] = self.all_agents_map[b] + new_loc_map
                # after motion, return reward for s'
                new_loc = torch.nonzero(new_loc_map, as_tuple=True)
                discovered_reward = self.grid[new_loc].item()
                self.reward[b] += discovered_reward
                if discovered_reward > 0.5:
                    self.terminal[b] = 1
        return self.reward.clone(), self.terminal.clone()

    def step(self, a_t):
        reward, terminal = self.get_reward(a_t)
        self.step_count += 1
        return reward, terminal

    def get_state(self, guide=False):
        if guide:
            return self.grid
        # non-guides only know their location
        return self.agent_locs

    def get_action_range(self):
        # only a function of the game
        bs = self.opt.bs
        # action range = int([0, 5))  0 = no action
        action_space_max = len(self.game_actions)
        comm_realval_space_max = action_space_max + 1 + 2 ** self.opt.game_comm_bits
        # define ranges per agent
        # pylint: disable=not-callable
        action_range = torch.tensor([0, action_space_max])
        comm_range = torch.tensor(
            [action_space_max, self.opt.game_action_space_total])
        # pylint: enable=not-callable
        # repeat for all agent batches
        action_range = action_range.repeat((bs, 1))
        comm_range = comm_range.repeat((bs, 1))
        return action_range, comm_range

    def _agents_will_collide(self, batch, curr_loc, proposed_action):
        agents_map = torch.nonzero(self.all_agents_map[batch], as_tuple=False)
        agent_locs_set = set()
        for n in agents_map:
            agent_locs_set.add((n[0].item(), n[1].item()))
        moves = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]
        proposed_y = curr_loc[0].item() + moves[proposed_action - 1][0]
        proposed_x = curr_loc[1].item() + moves[proposed_action - 1][1]
        return (proposed_y, proposed_x) in agent_locs_set

    def show(self, vid=True):
        if vid:
            os.system('cls' if os.name == 'nt' else "printf '\033c'")
        agent_locs = self.all_agents_map[0]
        for i in range(self.H):
            print('+' + ('-' * 9 + '+') * self.W)
            out = '| '
            for j in range(self.W):
                if agent_locs[i, j].item() == 1:
                    if i == self.ry and j == self.rx:
                        data = '*AGENT*'
                    else:
                        data = 'AGENT'
                else:
                    data = str(self.grid[i][j].item())
                out += data.ljust(7) + ' | '
            print(out)
        print('+' + ('-' * 9 + '+') * self.W)
        print()
