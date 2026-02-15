import os
import retro
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.wrappers import make_env


def train_stable_baselines(
    algorithm='PPO',
    num_timesteps=1000000,
    game_name='MortalKombat3-Genesis',
    state='Level1.Warrior.SubZeroVsScorpion',
    learning_rate=3e-4,
    n_steps=2048,
    gamma=0.99,
    ent_coef=0.01,
    save_freq=50000
):
    env = make_env(game_name=game_name, state=state)
    env = DummyVecEnv([lambda: env])
    
    os.makedirs('saved_models', exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path='./saved_models/',
        name_prefix=f'{algorithm.lower()}_mk3'
    )
    
    if algorithm == 'PPO':
        model = PPO(
            'CnnPolicy',
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log='./logs/'
        )
    elif algorithm == 'A2C':
        model = A2C(
            'CnnPolicy',
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log='./logs/'
        )
    else:
        raise ValueError(f'unknown algorithm: {algorithm}')
    
    print(f'training {algorithm} for {num_timesteps} timesteps...')
    model.learn(total_timesteps=num_timesteps, callback=checkpoint_callback)
    
    final_model_path = f'saved_models/{algorithm.lower()}_mk3_final'
    model.save(final_model_path)
    print(f'final model saved to {final_model_path}')
    
    env.close()


def hyperparameter_search():
    gamma_values = [0.95, 0.99, 0.995]
    ent_coef_values = [0.001, 0.01, 0.1]
    n_steps_values = [128, 512, 2048]
    lr_values = [1e-4, 3e-4, 1e-3]
    
    results = []
    
    for gamma in gamma_values:
        for ent_coef in ent_coef_values:
            for n_steps in n_steps_values:
                for lr in lr_values:
                    print(f'\ntesting: gamma={gamma}, ent_coef={ent_coef}, n_steps={n_steps}, lr={lr}')
                    
                    env = make_env()
                    env = DummyVecEnv([lambda: env])
                    
                    model = PPO(
                        'CnnPolicy',
                        env,
                        learning_rate=lr,
                        n_steps=n_steps,
                        gamma=gamma,
                        ent_coef=ent_coef,
                        verbose=0
                    )
                    
                    model.learn(total_timesteps=100000)
                    
                    eval_env = make_env()
                    obs = eval_env.reset()
                    total_reward = 0
                    
                    for _ in range(5000):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, info = eval_env.step(action)
                        total_reward += reward
                        if done:
                            obs = eval_env.reset()
                    
                    avg_reward = total_reward / 5000
                    results.append({
                        'gamma': gamma,
                        'ent_coef': ent_coef,
                        'n_steps': n_steps,
                        'lr': lr,
                        'avg_reward': avg_reward
                    })
                    
                    print(f'average reward: {avg_reward:.2f}')
                    
                    env.close()
                    eval_env.close()
    
    results.sort(key=lambda x: x['avg_reward'], reverse=True)
    
    print('\ntop 5 configurations:')
    for i, result in enumerate(results[:5]):
        print(f'{i+1}. gamma={result["gamma"]}, ent_coef={result["ent_coef"]}, '
              f'n_steps={result["n_steps"]}, lr={result["lr"]}, '
              f'avg_reward={result["avg_reward"]:.2f}')
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'A2C'])
    parser.add_argument('--timesteps', type=int, default=1000000)
    parser.add_argument('--hyperparameter_search', action='store_true')
    
    args = parser.parse_args()
    
    if args.hyperparameter_search:
        hyperparameter_search()
    else:
        train_stable_baselines(
            algorithm=args.algorithm,
            num_timesteps=args.timesteps
        )
