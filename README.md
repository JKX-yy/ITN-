# Task-Oriented Adaptive Learning of Robot Manipulation Skills

<div align="center">

[[Website]](https://jkx-yy.github.io/)
[[PDF] ](https://jkx-yy.github.io/ieeeconf__%E7%AC%AC%E4%B8%89%E7%89%88%E8%8B%B1%E6%96%87___8%E9%A1%B5_.pdf)



[![Python Version](https://img.shields.io/badge/Python-3.8-blue.svg)](https://github.com/eureka-research/Eureka)
[<img src="https://img.shields.io/badge/Framework-PyTorch-red.svg"/>](https://pytorch.org/)

______________________________________________________________________

![](image/ITS.png)        
</div>
<section class="section">
  <div class="container is-max-desktop">
    <div class="columns is-centered has-text-centered">
      <div class="column is-four-fifths">
        <h2 class="title is-3" align="center"> Abstract </h2>
        <div class="content has-text-justified">
          <p>
            The robot's operating environment and tasks are dynamically changing. Therefore, developing a mechanism that can infer other things from one fact to adapt to various scenarios is crucial for enhancing the robot's adaptive capabilities.  This paper proposes a general Intelligent Transfer System (ITS) to enable rapid skill learning for robots in dynamic tasks.ITS integrate Large Language Models (LLMs) with transfer learning, leveraging LLMs' intelligence and prior skill knowledge to expedite the learning of new skills.  It is capable of comprehending new and previously unseen task conmmands and automatically generating a process-oriented reward function for these task.This approach eliminates the need to design hierarchical sub-processes for  complex tasks. In addition, we designed an intelligent transfer network (ITN) in ITS to extract knowledge of relevant skills for the learning of new ones. This paper's method holds promise for enabling robots to independently solve entirely new operational tasks. We conducted a series of evaluations in a simulation environment, and the experimental results show that our method improves the time efficiency of two major tasks by 72.22\% and 65.17\%, respectively, compared with learning from scratch. Additionally, we compare our method with several outstanding works in the field.  You can get the code with all the experimental videos on our project website：https://jkx-yy.github.io
          </p>
        </div>
      </div>
    </div>
  </div>
</section>




# Installation
Eureka requires Python ≥ 3.8. We have tested on Ubuntu 20.04
1.clone
```
git clone https://github.com/JKX-yy/ITN-.git
```

2.  Creat Conda  Environment

```
conda create -n ITN python=3.8
conda activate ITN
```

3. Install IsaacGym (tested with `Preview Release 4/4`)
```	

cd ITN-/IsaacGym/isaacgym/python
pip install -e .
#(Test if the installation is successful and the emulation screen appears.)
cd ITN-/IsaacGym/isaacgym/python/examples
python joint_monkey.py
```

4. Install ITN
```
cd ITN-; pip install -e .
cd isaacgymenvs; pip install -e .
cd ../rl_games; pip install -e .
```

4. Using LLMs requires the use of the OpenAI API, and you need an OpenAI API key to use ITN.
   [here](https://platform.openai.com/account/api-keys)/   
```
openai.api_key= "YOUR_API_KEY"
```

# The Generator Generates the Reward Function

The generator generates the reward function,navigate to the `ITN` directory and run:
```
python ITN.py env={environment} iteration={num_iterations} sample={num_samples}
```
- `{environment}` is the task to perform. Options are listed in `ITN/cfg/env`.
- `{num_samples}` is the number of reward samples to generate per iteration. Default value is `10`.
- `{num_iterations}` is the number of  generator iterations to run. Default value is `5`.
You can set the default parameters in ITN/cfg/config.yaml, we have set the default task to factory_task_nut_bolt_pick_place. if you are using the default parameters then just run it in terminal:
```
python ITN.py 
```

Below are some example commands to try out ITN:
```
python ITN.py env=factory_task_nut_bolt_pick_place sample=10 iteration=2 model=gpt-4-0613
```
The results of the run can be viewed in ITN/outputs/ITN .You can also refer to https://github.com/eureka-research/Eureka. This project aims to further understand the principles of reward function generation, similar to our generator approach, except that the prompts and the problem to be solved are different. Our focus is on designing a process-oriented reward function for an  complex industrial operations problem.

# Robotics Skills Transfer Learning

1.We give the checkpoints model for several tasks at isaacgymenvs/isaacgymenvs/checkpoints. you can see the simulation directly by running the following commands in your terminal.
```
cd ITN/
```
FactoryTaskNutBoltPickGPT
```
python ./isaacgymenvs/isaacgymenvs/train.py test=True  is_transfor=False is_adapt_w=False headless=False force_render=True task=FactoryTaskNutBoltPickGPT checkpoint=isaacgymenvs/isaacgymenvs/checkpoints/last_FactoryTaskNutBoltPickGPT_ep_1024.pth

```
FactoryTaskNutBoltPickPlaceGPT

As long as the parameter is_transfor=True is_adapt_w=True. you need to open rl_games/rl_games/algos_torch/network_builder.py and comment out the default class A2CBuilder(NetworkBuilder) (# NO_TRANSFER). Free the class A2CBuilder(NetworkBuilder)# #ITN. In all other cases use default class A2CBuilder(NetworkBuilder). 

```
python ./isaacgymenvs/isaacgymenvs/train.py test=True  is_transfor=True is_adapt_w=True headless=False force_render=True task=FactoryTaskNutBoltPickPlaceGPT checkpoint=isaacgymenvs/isaacgymenvs/checkpoints/last_FactoryTaskNutBoltPickPlaceGPT_ep.pth
```

2.We have only given an example of transfer for the NutBolt_PickPlace task, the other tasks are on the same principle.

First of all the two base skills (Pick&Place) for this task are given in the skill space isaacgymenvs/isaacgymenvs/cfg/skill-space/skill-space.yaml, and you can also use only one of the skills for the target task for transfer learning.



We give several examples of basic skills training:
```
python ./isaacgymenvs/isaacgymenvs/train.py task=FactoryTaskNutBoltPlace  is_transfor=False  is_adapt_w=False   
```
```
python ./isaacgymenvs/isaacgymenvs/train.py task=FactoryTaskNutBoltPickGPT  is_transfor=False  is_adapt_w=False   
```
```
python ./isaacgymenvs/isaacgymenvs/train.py task=FactoryTaskNutBoltScrew  is_transfor=False  is_adapt_w=False   
```

We give running code for  complex skills No_Transfer and ITN learning:

Complex Skills  Learning from scratch (NO_Transfer)
```
python ./isaacgymenvs/isaacgymenvs/train.py task=FactoryTaskNutBoltPickPlaceGPT  is_transfor=False  is_adapt_w=False   
```
Complex Skills  Accelerated learning through transfer learning(ITN).

You need to open rl_games/rl_games/algos_torch/network_builder.py and comment out the default class A2CBuilder(NetworkBuilder) (# NO_TRANSFER). Free the class A2CBuilder(NetworkBuilder)# #ITN.

```
python ./isaacgymenvs/isaacgymenvs/train.py task=FactoryTaskNutBoltPickPlaceGPT  is_transfor=True  is_adapt_w=True 
```

# Creating  a New Environment
If you want to create a new task, you need to write a .yaml file and .py file, specifically you can the code in the project.

# Acknowledgement
We thank the following open-sourced projects:
- Our environments are from [IsaacGym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
- Our RL training code is based on [rl_games](https://github.com/Denys88/rl_games).
- Our partially generated code implementation references the  [eureka](https://github.com/eureka-research/Eureka).


