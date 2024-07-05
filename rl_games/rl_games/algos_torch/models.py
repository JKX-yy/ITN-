import rl_games.algos_torch.layers
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import rl_games.common.divergence as divergence
from rl_games.common.extensions.distributions import CategoricalMasked
from torch.distributions import Categorical
from rl_games.algos_torch.sac_helper import SquashedNormal
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl_games.algos_torch.moving_mean_std import GeneralizedMovingStats
from rl_games.algos_torch import torch_ext
checkpoint=[]

class BaseModel():
    def __init__(self, model_class):
        self.model_class = model_class

    def is_rnn(self):
        return False

    def is_separate_critic(self):
        return False

    def get_value_layer(self):
        return None

    def build(self, config):
        obs_shape = config['input_shape']
        normalize_value = config.get('normalize_value', False)
        normalize_input = config.get('normalize_input', False)
        value_size = config.get('value_size', 1)
        if isinstance(config['transfor'][0],str) :
            is_transfor=False
        else:
            is_transfor=True
        return self.Network(self.network_builder.build(self.model_class,**config), obs_shape=obs_shape,
            normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size,transfor=config['transfor'],is_transfor=is_transfor,)

class BaseModelNetwork(nn.Module):
    def __init__(self, obs_shape, normalize_value, normalize_input, value_size,transfor,is_transfor):
        nn.Module.__init__(self)
        self.obs_shape = obs_shape
        self.normalize_value = normalize_value
        self.normalize_input = normalize_input
        self.value_size = value_size

        if not is_transfor:
            if normalize_value:
                self.value_mean_std = RunningMeanStd((self.value_size,)) #   GeneralizedMovingStats((self.value_size,)) #   
            if normalize_input:
                if isinstance(obs_shape, dict):
                    self.running_mean_std = RunningMeanStdObs(obs_shape)
                else:
                    self.running_mean_std = RunningMeanStd(obs_shape)
        # transfer
        if  is_transfor:
            # checkpoint=[]
            for k in range(transfor[0].shape[1]):
                checkpoint.append(torch_ext.load_checkpoint(transfor[3][k])) 
            if normalize_value:
            #修改 记得该回去  
                for i in range(transfor[0].shape[1]):
                    setattr(self, f"value_mean_std{i}", RunningMeanStd((self.value_size,)))
                
                for i in range(transfor[0].shape[1]):
                    vms = getattr(self, f'value_mean_std{i}')
                    setattr(vms, 'count',  checkpoint[i]['model']['value_mean_std.count'])
                    setattr(vms, 'running_mean',  checkpoint[i]['model']['value_mean_std.running_mean'])
                    setattr(vms, 'running_var',  checkpoint[i]['model']['value_mean_std.running_var'])
                    vms_count= getattr(vms,'count')
                    vms_running_mean= getattr(vms,'running_mean')
                    vms_running_var= getattr(vms,'running_var')
                    setattr(vms_count, 'requires_grad', False)
                    setattr(vms_running_mean, 'requires_grad', False)
                    setattr(vms_running_var, 'requires_grad', False)
                                       
                    # setattr(self, f"value_mean_std{i}",nn.Parameter(checkpoint[i]['value_mean_std']['running_mean_std'], requires_grad=False))

                self.value_mean_std = RunningMeanStd((self.value_size,))
                
                
            if normalize_input:
                if isinstance(obs_shape, dict):
                    for i in range(transfor[0].shape[1]):
                        setattr(self, f"running_mean_std{i}", RunningMeanStdObs(obs_shape))
                    self.running_mean_std = RunningMeanStdObs(obs_shape)
                else:
                    for i in range(transfor[0].shape[1]):
                        setattr(self, f"running_mean_std{i}", RunningMeanStd(obs_shape))
                    for i in range(transfor[0].shape[1]):
                        rms = getattr(self, f'running_mean_std{i}')
                        # setattr(rms, 'training', False)
                        
                        setattr(rms, 'count',checkpoint[i]['model']['running_mean_std.count'])
                        setattr(rms, 'running_mean',checkpoint[i]['model']['running_mean_std.running_mean'])
                        setattr(rms, 'running_var',checkpoint[i]['model']['running_mean_std.running_var'])
                        # setattr(self, f"running_mean_std{i}",nn.Parameter(checkpoint[i]['model']['running_mean_std'], requires_grad=False))
                        rms_count= getattr(rms,'count')
                        rms_running_mean= getattr(rms,'running_mean')
                        rms_running_var= getattr(rms,'running_var')
                        setattr(rms_count, 'requires_grad', False)
                        setattr(rms_running_mean, 'requires_grad', False)
                        setattr(rms_running_var, 'requires_grad', False)
                    self.running_mean_std = RunningMeanStd(obs_shape)
                    
                    # self.running_mean_std = RunningMeanStd(obs_shape)

        
    
    def norm_obs(self,transfor, observation):

        with torch.no_grad():
            if isinstance(transfor[0], str): 
                return self.running_mean_std(observation) if self.normalize_input else observation
            elif isinstance(transfor[0], torch.Tensor): 
                obs=[]

                for i in range(transfor[0].shape[1]):
                    obs.append(getattr(self, f"running_mean_std{i}")(observation))
                obs.append(self.running_mean_std(observation) if self.normalize_input else observation)
                return obs
            
    def denorm_value(self,transfor,adapt_w,value):
        with torch.no_grad():
            return self.value_mean_std(value, denorm=True) if self.normalize_value else value
            
       
class ModelA2C(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self,**kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()
        
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()            

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def kl(self, p_dict, q_dict):
            p = p_dict['logits']
            q = q_dict['logits']
            return divergence.d_kl_discrete(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            action_masks = input_dict.get('action_masks', None)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            logits, value, states = self.a2c_network(input_dict)

            if is_train:
                categorical = CategoricalMasked(logits=logits, masks=action_masks)
                prev_neglogp = -categorical.log_prob(prev_actions)
                entropy = categorical.entropy()
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'logits' : categorical.logits,
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states
                }
                return result
            else:
                categorical = CategoricalMasked(logits=logits, masks=action_masks)
                selected_action = categorical.sample().long()
                neglogp = -categorical.log_prob(selected_action)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value),
                    'actions' : selected_action,
                    'logits' : categorical.logits,
                    'rnn_states' : states
                }
                return  result

class ModelA2CMultiDiscrete(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()
        
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def kl(self, p_dict, q_dict):
            p = p_dict['logits']
            q = q_dict['logits']
            return divergence.d_kl_discrete_list(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            action_masks = input_dict.get('action_masks', None)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            logits, value, states = self.a2c_network(input_dict)
            if is_train:
                if action_masks is None:
                    categorical = [Categorical(logits=logit) for logit in logits]
                else:   
                    categorical = [CategoricalMasked(logits=logit, masks=mask) for logit, mask in zip(logits, action_masks)]
                prev_actions = torch.split(prev_actions, 1, dim=-1)
                prev_neglogp = [-c.log_prob(a.squeeze()) for c,a in zip(categorical, prev_actions)]
                prev_neglogp = torch.stack(prev_neglogp, dim=-1).sum(dim=-1)
                entropy = [c.entropy() for c in categorical]
                entropy = torch.stack(entropy, dim=-1).sum(dim=-1)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'logits' : [c.logits for c in categorical],
                    'values' : value,
                    'entropy' : torch.squeeze(entropy),
                    'rnn_states' : states
                }
                return result
            else:
                if action_masks is None:
                    categorical = [Categorical(logits=logit) for logit in logits]
                else:   
                    categorical = [CategoricalMasked(logits=logit, masks=mask) for logit, mask in zip(logits, action_masks)]                
                
                selected_action = [c.sample().long() for c in categorical]
                neglogp = [-c.log_prob(a.squeeze()) for c,a in zip(categorical, selected_action)]
                selected_action = torch.stack(selected_action, dim=-1)
                neglogp = torch.stack(neglogp, dim=-1).sum(dim=-1)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value),
                    'actions' : selected_action,
                    'logits' : [c.logits for c in categorical],
                    'rnn_states' : states
                }
                return  result

class ModelA2CContinuous(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()
            
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def kl(self, p_dict, q_dict):
            p = p_dict['mu'], p_dict['sigma']
            q = q_dict['mu'], q_dict['sigma']
            return divergence.d_kl_normal(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            mu, sigma, value, states = self.a2c_network(input_dict)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)

            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = -distr.log_prob(prev_actions).sum(dim=-1)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'value' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result
            else:
                selected_action = distr.sample().squeeze()
                neglogp = -distr.log_prob(selected_action).sum(dim=-1)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value),
                    'actions' : selected_action,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return  result          


class ModelA2CContinuousLogStd(BaseModel):  
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network,transfor,is_transfor, **kwargs):
            BaseModelNetwork.__init__(self,transfor=transfor,is_transfor=is_transfor, **kwargs)
            self.a2c_network = a2c_network
            self.transfor=transfor
        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            distr_n=[]
            selected_action_n=[]
            sigma_n_exp=[]
            temp=[]
            # entropy_n=[]
            # prev_neglogp_n=[]
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(self.transfor,input_dict['obs'])  #27  8888
            mu, logstd, value, states,num_column,adapt_w,mu_n,sigma_n,value_n,adapt_weight_out = self.a2c_network(input_dict)  #网络输出  ######
            sigma = torch.exp(logstd)
            
          
        
            
            for i in range(num_column):
                if i !=num_column-1:
                    sigma_n_exp.append(torch.exp(sigma_n[i]))
                    distr_n.append(torch.distributions.Normal(mu_n[i], sigma_n_exp[i], validate_args=False))  #正态分布  均值 sigma
                    temp=distr_n[i].sample()
                    
                    selected_action_n.append(torch.tensor(temp).unsqueeze(dim=0))
            if num_column!=1:
                selected_action_n=torch.cat(selected_action_n,dim=0)
           
            distr = torch.distributions.Normal(mu, sigma, validate_args=False) 
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma,
                    'mu_n':mu_n,
                    'sigma_n':sigma_n,
                }            
                return result
            else:  #<-
                selected_action = distr.sample()   
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)  
                
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(self.transfor,adapt_w,value),
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma,
                    'num_column':num_column,
                    'adapt_w':adapt_w,
                    'mu_n':mu_n,
                    'sigma_n':sigma_n,
                    'selected_action_n':selected_action_n,
                    'adapt_weight_out':adapt_weight_out
                }
                return result

        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)


class ModelCentralValue(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def kl(self, p_dict, q_dict):
            return None # or throw exception?

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            value, states = self.a2c_network(input_dict)
            if not is_train:
                value = self.denorm_value(value)

            result = {
                'values': value,
                'rnn_states': states
            }
            return result



class ModelSACContinuous(BaseModel):

    def __init__(self, network):
        BaseModel.__init__(self, 'sac')
        self.network_builder = network
    
    class Network(BaseModelNetwork):
        def __init__(self, sac_network,**kwargs):
            BaseModelNetwork.__init__(self,**kwargs)
            self.sac_network = sac_network

        def critic(self, obs, action):
            return self.sac_network.critic(obs, action)

        def critic_target(self, obs, action):
            return self.sac_network.critic_target(obs, action)

        def actor(self, obs):
            return self.sac_network.actor(obs)
        
        def is_rnn(self):
            return False

        def forward(self, input_dict):
            is_train = input_dict.pop('is_train', True)
            mu, sigma = self.sac_network(input_dict)
            dist = SquashedNormal(mu, sigma)
            return dist



