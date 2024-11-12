# Pytorch Version for LIO
This repo is based on homo-lio code with pymarl framework. We adapt LIO from tf into torch in this code for further development on ID&MD and Arrival.

#### Update Version
- v0.1 2024.10.22 Add lio_agent.py, lio_mac.py, modify train.py of LIO in pymarl.

- v0.2 2024.10.28 Add lio_learner. 

- v0.3 2024.11.10 保证一致性，从 list commit 下一个版本改成 ind learning，在mac里维护一个agent list，修改 learner train 的逻辑


#### ToDo List
- lio_controller line 74/74 有问题
- lio_controller 里的 select reward incentive 函数没写
- 需要考虑新写一个lio runner来控制buffer和reward

#### Contributiors:
- Jianing Yanglu
- Dan





## Learning Homophilic Incentives in Sequential Social Dilemmas
Our method is built on [PyMARL](https://github.com/oxwhirl/pymarl) and Sequential Social Dilemma Games ([SSDG](https://github.com/eugenevinitsky/sequential_social_dilemma_games)). We use the implementation of Cleanup and Harvest Game from the SSDG repo. The implementation of our algorithm is based on independent Q-learning from the PyMARL repo.


## Installation instructions

Set up a virtual environment and install the necessary packages using `requirements.txt` file.

```bash
conda create -n SSD python=3.7
conda activate SSD
pip install -r requirements.txt
```

## Run an experiment 

```bash
python3 src/main.py 
--config=homophily 
--env-config=cleanup 
with 
env_args.map=default5
env_args.num_agents=5
use_tensorboard=True 
```

For Cleanup with 10 agents, set `env_args.map=default10` and `env_args.num_agents=10`.
For Cleanup with 3 agents, set `env_args.map=default3` and `env_args.num_agents=3`.
For Harvest, set `--env-config=harvest`.

The config files contain default settings for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`.
`--env-config` refers to the config files in `src/config/envs`.

All results will be stored in the `results` folder.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is `False` by default. The frequency of saving models can be adjusted using `save_model_interval`. Models will be saved in the result directory, under the folder called *models*. The directory corresponding to each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching replays

`save_replay` option allows saving replays of models which are loaded from `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run in the test mode. 

```bash
python3 src/main.py 
--config=homophily 
--env-config=cleanup 
with 
env_args.map=default5
env_args.num_agents=5
use_tensorboard=False 
save_replay=True 
env_args.is_replay=True 
test_nepisode=1 
checkpoint_path="results/models/model_file_name"
```

The replays can be found in `results/replays/` folder.
