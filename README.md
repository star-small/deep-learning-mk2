# mortal kombat 2 reinforcement learning project

train an ai agent to play mortal kombat 2 using reinforcement learning techniques (ppo, a2c).

## project structure

```
mk2_rl_project/
├── agents/
│   └── ppo_agent.py          # ppo agent implementation
├── models/
│   └── cnn_policy.py          # cnn policy network
├── preprocessing/
│   └── wrappers.py            # gym environment wrappers (mk2 version)
├── utils/
│   └── action_mapping.py      # action space utilities
├── saved_models/              # saved model checkpoints
├── logs/                      # training logs
├── requirements.txt           # python dependencies
├── train_ppo.py              # main ppo training script (mk2)
├── train_stable_baselines.py # stable baselines training
├── evaluate.py               # model evaluation script (mk2)
└── test_setup.py             # setup verification (mk2)
```

## setup

### install dependencies

```bash
pip install -r requirements.txt
```

### setup gym retro

1. import mortal kombat 2 rom:

```bash
python -m retro.import /home/troy/dev/deep-learning-mk3/ROM/
```

2. verify installation:

```bash
python -c "import retro; env = retro.make(game='MortalKombatII-Genesis'); print('success')"
```

## usage

### train with custom ppo agent

```bash
python train_ppo.py
```

this will train for 2500 episodes by default. models are saved every 100 episodes to `saved_models/`.

### evaluate trained model

```bash
python evaluate.py --model_path saved_models/ppo_mk2_episode_2500.pth --num_episodes 10
```

### test setup

```bash
python test_setup.py
```

## training details

### preprocessing
- grayscale conversion
- downsampling to 84x84
- frame skipping (every 3 frames)

### reward function
```
reward = 1.0 * health_diff - 1.0 * enemy_health_diff + 10.0 * rounds_won
```

note: mk2 has 200 health (vs mk3's 176), so the reward wrapper is adjusted accordingly.

### model architecture
- 5 convolutional layers with relu activation
- 1 fully connected layer (512 units)
- 2 output heads: movement actions (5) and attack actions (7)
- xavier uniform initialization

### hyperparameters (ppo)
- learning rate: 3e-4
- gamma: 0.99
- gae lambda: 0.95
- clip epsilon: 0.2
- value loss coefficient: 0.5
- entropy coefficient: 0.01

## differences from mk3

1. **health values**: mk2 uses 200 health vs mk3's 176
2. **game name**: 'MortalKombatII-Genesis' instead of 'MortalKombat3-Genesis'
3. **default state**: 'Level1.SubZeroVsScorpion' (you can change this to other character matchups)

## available states

mk2 has different character matchups you can use for training. to see available states:

```bash
python -c "import retro; print(retro.data.list_states('MortalKombatII-Genesis'))"
```

## expected results

- after 600 episodes: average reward ~300
- after 1800 episodes: average reward ~442
- after 2500 episodes: can consistently beat level 1 difficulty
- learned behaviors: combos, ranged attacks, blocking

## troubleshooting

### rom not found
make sure you've imported the rom correctly:
```bash
python -m retro.import /home/troy/dev/deep-learning-mk3/ROM/
```

### out of memory
reduce batch size or use cpu instead of gpu by setting device='cpu' in training scripts.

### slow training
- ensure cuda is available if using gpu
- reduce image size in wrappers.py
- increase frame skip value

## files for mk2

use these mk2-specific files:
- wrappers_mk2.py (rename to wrappers.py in preprocessing/)
- train_ppo_mk2.py (rename to train_ppo.py in root)
- evaluate_mk2.py (rename to evaluate.py in root)
- test_setup_mk2.py (rename to test_setup.py in root)

all other files (cnn_policy.py, ppo_agent.py, action_mapping.py, etc.) work for both mk2 and mk3.

## references

based on the medium article:
https://medium.com/@zdwempe/using-reinforcement-learning-to-play-mortal-kombat-3-4f7e8bba7ab5

adapted for mortal kombat 2.
