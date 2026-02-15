import os
import sys
import numpy as np
import torch
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.wrappers import make_env
from models.cnn_policy import CNNPolicy
from agents.ppo_agent import PPOAgent
from utils.action_mapping import create_action_from_indices, buttons_to_action_array, get_retro_button_list


def train(
    num_episodes=2500,
    max_steps_per_episode=5000,
    update_interval=2048,
    save_interval=100,
    log_interval=10,
    game_name='MortalKombat3-Genesis',
    state='Level1.Warrior.SubZeroVsScorpion'
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
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.1,
        device=device
    )
    
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_{timestamp}.txt'
    
    episode_rewards = []
    total_steps = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done and episode_steps < max_steps_per_episode:
            movement_action, attack_action, value, movement_log_prob, attack_log_prob = agent.select_action(state)
            
            buttons = create_action_from_indices(movement_action, attack_action)
            action_array = buttons_to_action_array(buttons, button_list)
            
            next_state, reward, done, info = env.step(action_array)
            
            agent.store_transition(
                state, movement_action, attack_action, 
                reward, done, value, 
                movement_log_prob, attack_log_prob
            )
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if total_steps % update_interval == 0 and len(agent.rollout_buffer['states']) > 0:
                metrics = agent.update(state)
                
                if episode % log_interval == 0:
                    print(f'episode {episode}, step {total_steps}: '
                          f'policy_loss={metrics["policy_loss"]:.4f}, '
                          f'value_loss={metrics["value_loss"]:.4f}, '
                          f'entropy={metrics["entropy"]:.4f}')
        
        if len(agent.rollout_buffer['states']) > 0:
            metrics = agent.update(state)
        
        episode_rewards.append(episode_reward)
        
        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            print(f'episode {episode}/{num_episodes}, '
                  f'reward: {episode_reward:.2f}, '
                  f'avg_reward (last {log_interval}): {avg_reward:.2f}, '
                  f'steps: {episode_steps}')
            
            with open(log_file, 'a') as f:
                f.write(f'{episode},{episode_reward},{avg_reward},{episode_steps}\n')
        
        if episode % save_interval == 0 and episode > 0:
            save_path = f'saved_models/ppo_mk3_episode_{episode}.pth'
            agent.save(save_path)
            print(f'model saved to {save_path}')
    
    final_save_path = f'saved_models/ppo_mk3_final_{timestamp}.pth'
    agent.save(final_save_path)
    print(f'final model saved to {final_save_path}')
    
    env.close()
    print('training complete')


if __name__ == '__main__':
    train()
