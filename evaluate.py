import os
import sys
import torch
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.wrappers import make_env
from models.cnn_policy import CNNPolicy
from agents.ppo_agent import PPOAgent
from utils.action_mapping import create_action_from_indices, buttons_to_action_array, get_retro_button_list


def evaluate_model(
    model_path,
    num_episodes=10,
    max_steps_per_episode=5000,
    render=True,
    fps=60,
    game_name='MortalKombatII-Genesis',
    state='Level1.LiuKangVsJax'
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device: {device}')
    
    env = make_env(game_name=game_name, state=state)
    button_list = get_retro_button_list()
    
    num_movement_actions = 5
    num_attack_actions = 7
    
    policy = CNNPolicy(
        input_channels=1,
        num_movement_actions=num_movement_actions,
        num_attack_actions=num_attack_actions
    )
    
    agent = PPOAgent(
        policy_network=policy,
        num_movement_actions=num_movement_actions,
        num_attack_actions=num_attack_actions,
        device=device
    )
    
    agent.load(model_path)
    print(f'loaded model from {model_path}')
    
    episode_rewards = []
    episode_lengths = []
    wins = 0
    
    frame_time = 1.0 / fps if fps > 0 else 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        print(f'\nstarting episode {episode + 1}/{num_episodes}...')
        
        while not done and episode_steps < max_steps_per_episode:
            start_time = time.time()
            
            movement_action, attack_action, _, _, _ = agent.select_action(state)
            
            buttons = create_action_from_indices(movement_action, attack_action)
            action_array = buttons_to_action_array(buttons, button_list)
            
            state, reward, done, info = env.step(action_array)
            
            if render:
                env.render()
            
            episode_reward += reward
            episode_steps += 1
            
            # frame rate limiting
            if frame_time > 0:
                elapsed = time.time() - start_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        
        if info.get('rounds_won', 0) >= 2:
            wins += 1
        
        print(f'episode {episode + 1}/{num_episodes}: '
              f'reward={episode_reward:.2f}, '
              f'steps={episode_steps}, '
              f'rounds_won={info.get("rounds_won", 0)}')
    
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    win_rate = wins / num_episodes
    
    print(f'\nevaluation results over {num_episodes} episodes:')
    print(f'average reward: {avg_reward:.2f}')
    print(f'average episode length: {avg_length:.2f}')
    print(f'win rate: {win_rate:.2%}')
    
    env.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--no_render', action='store_true')
    parser.add_argument('--fps', type=int, default=60, help='frames per second (0 for unlimited)')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        render=not args.no_render,
        fps=args.fps
    )
