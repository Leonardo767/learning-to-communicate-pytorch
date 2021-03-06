import copy
import argparse
import csv
import json
import datetime
import os
from functools import partial
from pathlib import Path

from utils.dotdic import DotDic
from arena import Arena
from agent import CNetAgent
from envs.cnet import CNet
import torch

"""
Play communication games
"""

# configure opts for Switch game with 3 DIAL agents


def init_action_and_comm_bits(opt):
    opt.comm_enabled = opt.game_comm_bits > 0  # and opt.game_nagents > 1
    if opt.model_comm_narrow is None:
        opt.model_comm_narrow = opt.model_dial
    if not opt.model_comm_narrow and opt.game_comm_bits > 0:
        opt.game_comm_bits = 2 ** opt.game_comm_bits
    if opt.comm_enabled:
        opt.game_action_space_total = opt.game_action_space + opt.game_comm_bits
    else:
        opt.game_action_space_total = opt.game_action_space
    return opt


def init_opt(opt):
    if not opt.model_rnn_layers:
        opt.model_rnn_layers = 2
    if opt.model_avg_q is None:
        opt.model_avg_q = True
    if opt.eps_decay is None:
        opt.eps_decay = 1.0
    opt = init_action_and_comm_bits(opt)
    opt.game = opt.game.lower()
    return opt


def create_game(opt):
    game_name = opt.game
    if game_name == 'switch':
        from envs.switch_game import SwitchGame
        return SwitchGame(opt)
    elif game_name == 'grid':
        from envs.grid_game_flat import GridGame
        return GridGame(opt, (4, 4))
    else:
        raise Exception('Unknown game: {}'.format(game_name))


def create_cnet(opt):
    game_name = opt.game
    if game_name == 'switch':
        return CNet(opt)
    elif game_name == 'grid':
        return CNet(opt)
    else:
        raise Exception('Unknown game: {}'.format(game_name))


def create_agents(opt, game):
    agents = [None]  # 1-index agents
    cnet = create_cnet(opt)
    cnet_target = copy.deepcopy(cnet)
    for i in range(1, opt.game_nagents + 1):
        agents.append(CNetAgent(opt, game=game, model=cnet,
                                target=cnet_target, index=i))
        if not opt.model_know_share:
            cnet = create_cnet(opt)
            cnet_target = copy.deepcopy(cnet)
    return agents


def save_episode_and_reward_to_csv(file, writer, e, r):
    writer.writerow({'episode': e, 'reward': r})
    file.flush()


def run_trial(opt, result_path=None, verbose=False):
    # Initialize action and comm bit settings
    opt = init_opt(opt)

    game = create_game(opt)
    agents = create_agents(opt, game)
    arena = Arena(opt, game)

    test_callback = None
    if result_path:
        result_out = open(result_path, 'w')
        csv_meta = '#' + json.dumps(opt) + '\n'
        result_out.write(csv_meta)
        writer = csv.DictWriter(result_out, fieldnames=['episode', 'reward'])
        writer.writeheader()
        test_callback = partial(
            save_episode_and_reward_to_csv, result_out, writer)
    arena.train(agents, verbose=verbose, test_callback=test_callback)

    if result_path:
        result_out.close()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    parsing = False
    if parsing:
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config_path', type=str,
                            help='path to existing options file')
        parser.add_argument('-r', '--results_path', type=str,
                            help='path to results directory')
        parser.add_argument('-n', '--ntrials', type=int,
                            default=1, help='number of trials to run')
        parser.add_argument('-s', '--start_index', type=int,
                            default=0, help='starting index for trial output')
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='prints training epoch rewards if set')
        args = parser.parse_args()
        config_path = args.config_path
        ntrials = args.ntrials
        start_idx = args.start_index
        result_path = args.results_path
        verbose = args.verbose
    else:
        config_path = 'config/grid_3_dial.json'
        ntrials = 1
        start_index = 0
        result_path = False
        verbose = False

    opt = DotDic(json.loads(open(config_path, 'r').read()))

    # if args.results_path:
    #     result_path = args.config_path and os.path.join(Path(args.config_path).stem, args.results_path) or \
    #         os.path.join(args.results_path, 'result-',
    #                      datetime.datetime.now().isoformat())
    #     print(result_path)

    for i in range(ntrials):
        trial_result_path = None
        if result_path:
            trial_result_path = result_path + '_' + \
                str(i + start_index) + '.csv'
        trial_opt = copy.deepcopy(opt)
        run_trial(trial_opt, result_path=trial_result_path,
                  verbose=verbose)
