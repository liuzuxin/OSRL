# Decision Transformer for Offline Safe Reinforcement Learning

## Installation
Pull the repo and init all the submodules
```
git clone git@github.com:liuzuxin/offline-safe-rl.git
cd offline-safe-rl
git submodule update --init --recursive
```

Create an anaconda environment:
```
conda create -n offline python=3.8
conda activate offline
```
Install the `bullet_safety_gym` simulation environment and the `saferl` library:
```
cd Bullet-Safety-Gym
pip install -e .
cd ../safe-rl-lib
pip install -e .
```

Finally install the `d4srl` library and `sdt` library:
```
cd ../d4srl
pip install -e .
cd ../
pip install SWIG
pip install -e .
```

## Structure
The structure of this repo is as follows:
```
├── baselines  # offline safe RL baselines
├── Bullet-Safety-Gym  # experiment environment
├── safe-rl-lib  # safe RL expert libraries
├── d4srl  # a wrapper implementation for offline safe RL
├── dataset  # a wrapper implementation for offline safe RL
│   ├── data # store .hdf5 dataset, use git lfs to manage
│   ├── train_xxx.py # use sac or ppo expert to collect offline data
│   ├── create_dataset.py # process the saved offline data to d4rl format
│   ├── stitcb_dataset.py # stitch multiple datasets together
├── sdt  # implementation for safe DT
├── config  # configs for safe DT
├── run.py  # train safe DT models
├── eval.py # evaluate trained safe DT model
```

## Create the dataset

1. Running the `sac-lagrangian` algorithm to collect the entire replay buffer by the following command:
```
cd dataset
python train_sac.py --config sac_config/cc.yaml no_save=1
```
where `--config` specify the configuration parameters. Setting `no_save` mode to 0 will save the replay buffer at the end, 
otherwise it will not save anything. The default param for `no_save` is 0.

You can also use the `ppo-lagrangian` algorithm, which might have better perf in `drone` tasks. Some comments about the hyper-parameters are in `ppo_config/cc.yaml` folder.

When creating dataset for a new environment, you should tune the hyper-parameters so set `no_save` to 1 will be better.

2. Process the saved replay buffer as the `d4rl` format by running the following script:
```
python create_dataset.py /path/to/saved/buffer -o /name/of/created/dataset
```
For example:
```
python create_dataset.py log/SafetyCarCircle-v0/sac_cost80/dataset.hdf5 --output cc-cost80-replay.hdf5
```

## Train safe DT
The detailed configurations are in `run.py`. To train, simply run:
```
python run.py --env_name SafetyCarCircle-v0 --dataset ./dataset/data/cc-cost1.hdf5 --other-arguments
```
It will automatically generate experiment name if you modify some parameters. 

The default path to save the model is in `./log/exp-name/checkpoint`.


## Evaluate trained safe DT model

To evaluate a trained model, run:
```
python eval.py  --path /path/to/model.pt --returns [300, 400] --costs [0, 0]
```


## TODO

1. 采集数据的script再改下，对每个环境还是改成那种decay cost limit的，尽可能采集到更多PF上的点。
2. 对采集到的数据做降采样，但尽量保持数据点分布均匀且PF上的点得保留。我预计数据点少的话，baseline的效果不会很好。
3. 降采样完之后，根据设定的cost limit范围选取那对应范围的数据，就当作这个环境的数据集了。比如cc-cost-from-5-to-40.hdf5
4. 做数据增强，采样PF上面的reward和cost，然后取左下角PF附近的trajectory当作数据点。
5. 根据你现在写的这个fit curve和算distance的来算增强后的数据集的每条traj weight，用来训练SDT
