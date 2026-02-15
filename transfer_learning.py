import os
import sys
import torch
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.wrappers import make_env
from models.cnn_policy import CNNPolicy
from agents.ppo_agent import PPOAgent
from utils.action_mapping import create_action_from_indices, buttons_to_action_array, get_retro_button_list


def transfer_learning(
    pretrained_model_path,
    target_game='MortalKombat3-Genesis',
    target_state='Level1.Warrior.SubZeroVsScorpion',
    num_episodes=100,
    freeze_conv_layers=False,
    reinit_fc=False
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device: {device}')
    
    env = make_env(game_name=target_game, state=target_state)
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
    
    agent.load(pretrained_model_path)
    print(f'loaded pretrained model from {pretrained_model_path}')
    
    if freeze_conv_layers:
        for param in agent.policy.conv1.parameters():
            param.requires_grad = False
        for param in agent.policy.conv2.parameters():
            param.requires_grad = False
        for param in agent.policy.conv3.parameters():
            param.requires_grad = False
        for param in agent.policy.conv4.parameters():
            param.requires_grad = False
        for param in agent.policy.conv5.parameters():
            param.requires_grad = False
        print('froze convolutional layers')
    
    if reinit_fc:
        torch.nn.init.xavier_uniform_(agent.policy.fc.weight)
        torch.nn.init.constant_(agent.policy.fc.bias, 0)
        torch.nn.init.xavier_uniform_(agent.policy.movement_head.weight)
        torch.nn.init.constant_(agent.policy.movement_head.bias, 0)
        torch.nn.init.xavier_uniform_(agent.policy.attack_head.weight)
        torch.nn.init.constant_(agent.policy.attack_head.bias, 0)
        print('reinitialized fully connected layers')
    
    agent.optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, agent.policy.parameters()),
        lr=3e-4
    )
    
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/transfer_{timestamp}.txt'
    
    episode_rewards = []
    total_steps = 0
    update_interval = 2048
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done and episode_steps < 5000:
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
        
        if len(agent.rollout_buffer['states']) > 0:
            metrics = agent.update(state)
        
        episode_rewards.append(episode_reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f'episode {episode}/{num_episodes}, '
                  f'reward: {episode_reward:.2f}, '
                  f'avg_reward (last 10): {avg_reward:.2f}')
            
            with open(log_file, 'a') as f:
                f.write(f'{episode},{episode_reward},{avg_reward}\n')
        
        if episode % 50 == 0 and episode > 0:
            save_path = f'saved_models/transfer_episode_{episode}.pth'
            agent.save(save_path)
            print(f'model saved to {save_path}')
    
    final_save_path = f'saved_models/transfer_final_{timestamp}.pth'
    agent.save(final_save_path)
    print(f'final model saved to {final_save_path}')
    
    env.close()
    print('transfer learning complete')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, required=True)
    parser.add_argument('--target_game', type=str, default='MortalKombat3-Genesis')
    parser.add_argument('--target_state', type=str, default='Level1.Warrior.SubZeroVsScorpion')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--freeze_conv', action='store_true')
    parser.add_argument('--reinit_fc', action='store_true')
    
    args = parser.parse_args()
    
    transfer_learning(
        pretrained_model_path=args.pretrained_model,
        target_game=args.target_game,
        target_state=args.target_state,
        num_episodes=args.num_episodes,
        freeze_conv_layers=args.freeze_conv,
        reinit_fc=args.reinit_fc
    )
