import os
import time
import numpy as np
import random
from copy import deepcopy
import torch
from rl_games.algos_torch import torch_ext
#import yaml

#from rl_games import envs
from rl_games.common import object_factory
from rl_games.common import tr_helpers

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import a2c_discrete
from rl_games.algos_torch import players
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.algos_torch import sac_agent

def _restore(agent, args):
    if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
        agent.restore(args['checkpoint'])

def _override_sigma(agent, args):
    if 'sigma' in args and args['sigma'] is not None:
        net = agent.model.a2c_network
        if hasattr(net, 'sigma') and hasattr(net, 'fixed_sigma'):
            if net.fixed_sigma:
                with torch.no_grad():
                    net.sigma.fill_(float(args['sigma']))
            else:
                print('Print cannot set new sigma because fixed_sigma is False')


class Runner:
    def __init__(self, algo_observer=None):
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.DiscreteA2CAgent(**kwargs)) 
        self.algo_factory.register_builder('sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))
        #self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs))
        self.player_factory.register_builder('sac', lambda **kwargs : players.SACPlayer(**kwargs))
        #self.player_factory.register_builder('dqn', lambda **kwargs : players.DQNPlayer(**kwargs))

        self.algo_observer = algo_observer if algo_observer else DefaultAlgoObserver()
        torch.backends.cudnn.benchmark = True
        ### it didnot help for lots for openai gym envs anyway :(
        #torch.backends.cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)

    def reset(self):
        pass

    def load_config(self, params):
        self.seed = params.get('seed', None)
        if self.seed is None:
            self.seed = int(time.time())

        if params["config"].get('multi_gpu', False):
            self.seed += int(os.getenv("LOCAL_RANK", "0"))
        print(f"self.seed = {self.seed}")

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.exp_config = None

        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

            # deal with environment specific seed if applicable
            if 'env_config' in params['config']:
                if not 'seed' in params['config']['env_config']:
                    params['config']['env_config']['seed'] = self.seed
                else:
                    if params["config"].get('multi_gpu', False):
                        params['config']['env_config']['seed'] += int(os.getenv("LOCAL_RANK", "0"))

        config = params['config']
        config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
        if 'features' not in config:
            config['features'] = {}
        config['features']['observer'] = self.algo_observer
        self.params = params

    def load(self, yaml_config):
        config = deepcopy(yaml_config)
        self.default_config = deepcopy(config['params'])
        self.load_config(params=self.default_config)

    def run_train(self, args):
        print('Started to train')
        # self.params.add()
        # self.transfor=args['transfor']
        self.params['transfor']=args['transfor']
        self.params['config']['model_gpt']=args['model_gpt']
        self.params['config']['gpt_key']=args['gpt_key']
        self.params['config']['gpt_url']=args['gpt_url']
        self.params['config']['init_transfor_sys_path']=args['init_transfor_sys_path']

        self.agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)  #初始化模型
        # if  args['is_adapt_w']:
        #     checkpoint = torch_ext.load_checkpoint(args['transfor'][3][0])
        #     # self.agent.model.running_mean_std.load_state_dict(checkpoint['model']['running_mean_std'])
        #     self.agent.model.running_mean_std.count=checkpoint['model']['running_mean_std.count']
        #     self.agent.model.running_mean_std.running_mean=checkpoint['model']['running_mean_std.running_mean']
        #     self.agent.model.running_mean_std.running_var=checkpoint['model']['running_mean_std.running_var']
        # 迁移什么时候赋予不同的人物的  std 什么时候是不迁移的std  不迁移的std是怎么用的 怎么富裕的。
        
        _restore(self.agent, args) #没看
        _override_sigma(self.agent, args)  #none  直接返回不满足if
        self.agent.transfor=args['transfor']
        self.agent.is_adapt_w=args['is_adapt_w']
        self.agent.is_soft_attention=args['is_soft_attention']
        self.agent.adapt_fps=args['adapt_fps']
        self.agent.train()

    def run_play(self, args):
        # args['checkpoint']='/home/jkx/桌面/project/ITN/isaacgymenvs/isaacgymenvs/checkpoints/last_FactoryTaskNutBoltPickGPT_ep_1024.pth'
        print('Started to play')
        player = self.create_player(args)
        _restore(player, args)  #   running_mean_std
        _override_sigma(player, args)
        player.run(args)

    def create_player(self,args):
        self.params['transfor']=args['transfor']
        return self.player_factory.create(self.algo_name, params=self.params)
    
    def restore_player(self, player, args):
        _restore(player, args)
        _override_sigma(player, args)

    def reset(self):
        pass

    def run(self, args):
        load_path = None

        if args['train']:
            self.run_train(args)

        elif args['play']:
            self.run_play(args)
        else:
            self.run_train(args)
        # data = load_tensorboard_logs(self.agent.writer.logdir)
        # return data 