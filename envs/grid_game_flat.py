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
                "Config action space doesn't  match game's ({} != {}).".format(
                    self.opt.game_action_space, len(self.game_actions)))

        self.H = size[0]
        self.W = size[1]
        self.dim = self.H * self.W
        self.goal_reward = 10
        self.reset()

    def reset(self):
        # episode flags
        self.step_count = 0
        self.reward = torch.zeros(self.opt.bs, self.opt.game_nagents)
        self.terminal = torch.zeros(self.opt.bs, dtype=torch.long)
        # init common reward one-hot (dim)
        self.grid = torch.zeros(self.dim)
        self.reward_location = np.random.randint(0, self.dim)
        # init agent locations (bs x n x dim)
        bs = self.opt.bs
        n_agents = self.opt.game_nagents
        if n_agents > self.dim:
            raise ValueError("Too many agents to fit inside grid.")
        self.agent_locs = torch.zeros(
            bs, n_agents, self.dim, dtype=torch.long)
        self.all_agents_map = torch.zeros(bs, self.dim)
        for b in range(bs):
            # per batch, build world of n_agents, shuffle them around the grid
            agent_world = torch.zeros(self.dim)
            agent_world[0:n_agents] = torch.ones(n_agents)
            idx = torch.randperm(agent_world.nelement())
            agent_world = agent_world[idx]
            self.all_agents_map[b] = agent_world  # shuffled agent map
            locs_for_batch = torch.nonzero(agent_world, as_tuple=False)
            for n in range(n_agents):
                self.agent_locs[b, n, locs_for_batch[n].item()] = 1

    def get_reward(self, a_t):
        # all actions cost -1 reward
        for b in range(self.opt.bs):
            self.reward[b] = self.reward[b] - 1
            for n in range(self.opt.game_nagents):
                # assess current location
                curr_loc = torch.argmax(self.agent_locs[b, n, :]).item()
                on_left_edge = curr_loc % self.W == 0
                on_right_edge = (curr_loc + 1) % self.W == 0
                on_top_edge = curr_loc < self.W
                on_bottom_edge = curr_loc + self.W >= self.dim
                proposed_action = int(a_t[b, n].item())
                # find agent [b, n] next location based on action
                if proposed_action == self.game_actions.UP and not on_top_edge:
                    next_loc = curr_loc - self.W
                elif proposed_action == self.game_actions.DOWN and not on_bottom_edge:
                    next_loc = curr_loc + self.W
                elif proposed_action == self.game_actions.LEFT and not on_left_edge:
                    next_loc = curr_loc - 1
                elif proposed_action == self.game_actions.RIGHT and not on_right_edge:
                    next_loc = curr_loc + 1
                else:
                    next_loc = curr_loc
                # check if square is unoccupied
                if self.all_agents_map[b, next_loc] == 0:
                    # update locations
                    self.all_agents_map[b, curr_loc] = 0
                    self.all_agents_map[b, next_loc] = 1
                    self.agent_locs[b, n, curr_loc] = 0
                    self.agent_locs[b, n, next_loc] = 1
                # after motion, return reward for s'
                discovered_reward = self.grid[next_loc].item()
                if next_loc == self.reward_location:
                    self.reward[b] = self.reward[b] + self.goal_reward
                    self.terminal[b] = 1
        return self.reward.clone(), self.terminal.clone()

    def step(self, a_t):
        reward, terminal = self.get_reward(a_t)
        self.step_count += 1
        return reward, terminal

    def get_state(self, guide=False):
        state = torch.zeros(
            self.opt.bs, self.opt.game_nagents, dtype=torch.long)
        # if guide:
        #     return self.grid
        # # non-guides only know their location
        for b in range(self.opt.bs):
            for n in range(self.opt.game_nagents):
                state[b, n] = torch.argmax(self.agent_locs[b, n])
        return state

    def get_action_range(self, step, agent_id):
        # only a function of the game
        bs = self.opt.bs
        action_dtype = torch.long
        # action range = int([0, 5))  0 = no action
        action_space_max = len(self.game_actions)
        comm_realval_space_max = action_space_max + 1 + 2 ** self.opt.game_comm_bits
        # define ranges per agent
        # pylint: disable=not-callable
        action_range = torch.tensor([0, action_space_max], dtype=action_dtype)
        comm_range = torch.tensor(
            [action_space_max, self.opt.game_action_space_total], dtype=action_dtype)
        # pylint: enable=not-callable
        # repeat for all agent batches
        action_range = action_range.repeat((bs, 1))
        comm_range = comm_range.repeat((bs, 1))
        return action_range, comm_range

    def get_comm_limited(self, step, agent_id):
        return torch.ones(self.opt.bs, dtype=torch.long)

    def get_stats(self, episode_steps):
        return 0

    def show(self, vid=True):
        if vid:
            os.system('cls' if os.name == 'nt' else "printf '\033c'")
        agent_locs = self.all_agents_map[0]
        grid_idx = -1
        for i in range(self.H):
            print('+' + ('-' * 9 + '+') * self.W)
            out = '| '
            for j in range(self.W):
                grid_idx += 1
                if agent_locs[grid_idx].item() == 1:
                    if grid_idx == self.reward_location:
                        data = '*AGENT*'
                    else:
                        data = 'AGENT'
                elif grid_idx == self.reward_location:
                    data = str(self.goal_reward)
                else:
                    data = ' '
                out += data.ljust(7) + ' | '
            print(out)
        print('+' + ('-' * 9 + '+') * self.W)
        print()
