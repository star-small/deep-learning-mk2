import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class PPOAgent:
    def __init__(
        self, 
        policy_network, 
        num_movement_actions,
        num_attack_actions,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.policy = policy_network.to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.num_movement_actions = num_movement_actions
        self.num_attack_actions = num_attack_actions
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.rollout_buffer = {
            'states': [],
            'movement_actions': [],
            'attack_actions': [],
            'rewards': [],
            'values': [],
            'dones': [],
            'movement_log_probs': [],
            'attack_log_probs': []
        }
    
    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            if len(state_tensor.shape) == 4:
                # shape is (1, 84, 84, 1), need to transpose to (1, 1, 84, 84)
                state_tensor = state_tensor.permute(0, 3, 1, 2)
            
            movement_probs, attack_probs, value = self.policy(state_tensor)
            
            movement_dist = torch.distributions.Categorical(movement_probs)
            attack_dist = torch.distributions.Categorical(attack_probs)
            
            movement_action = movement_dist.sample()
            attack_action = attack_dist.sample()
            
            movement_log_prob = movement_dist.log_prob(movement_action)
            attack_log_prob = attack_dist.log_prob(attack_action)
            
        return (
            movement_action.item(), 
            attack_action.item(),
            value.item(),
            movement_log_prob.item(),
            attack_log_prob.item()
        )
    
    def store_transition(self, state, movement_action, attack_action, reward, done, value, movement_log_prob, attack_log_prob):
        self.rollout_buffer['states'].append(state)
        self.rollout_buffer['movement_actions'].append(movement_action)
        self.rollout_buffer['attack_actions'].append(attack_action)
        self.rollout_buffer['rewards'].append(reward)
        self.rollout_buffer['dones'].append(done)
        self.rollout_buffer['values'].append(value)
        self.rollout_buffer['movement_log_probs'].append(movement_log_prob)
        self.rollout_buffer['attack_log_probs'].append(attack_log_prob)
    
    def compute_gae(self, next_value):
        rewards = np.array(self.rollout_buffer['rewards'])
        values = np.array(self.rollout_buffer['values'] + [next_value])
        dones = np.array(self.rollout_buffer['dones'])
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values[:-1]
        return advantages, returns
    
    def update(self, next_state, epochs=4, batch_size=32):
        with torch.no_grad():
            next_state_tensor = torch.from_numpy(next_state).unsqueeze(0).to(self.device)
            if len(next_state_tensor.shape) == 4:
                # shape is (1, 84, 84, 1), need to transpose to (1, 1, 84, 84)
                next_state_tensor = next_state_tensor.permute(0, 3, 1, 2)
            _, _, next_value = self.policy(next_state_tensor)
            next_value = next_value.item()
        
        advantages, returns = self.compute_gae(next_value)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        states = torch.from_numpy(np.array(self.rollout_buffer['states'])).to(self.device)
        # transpose states from (N, 84, 84, 1) to (N, 1, 84, 84)
        if len(states.shape) == 4 and states.shape[-1] == 1:
            states = states.permute(0, 3, 1, 2)
        movement_actions = torch.LongTensor(self.rollout_buffer['movement_actions']).to(self.device)
        attack_actions = torch.LongTensor(self.rollout_buffer['attack_actions']).to(self.device)
        old_movement_log_probs = torch.FloatTensor(self.rollout_buffer['movement_log_probs']).to(self.device)
        old_attack_log_probs = torch.FloatTensor(self.rollout_buffer['attack_log_probs']).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for epoch in range(epochs):
            np.random.shuffle(indices)
            
            for start_idx in range(0, dataset_size, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                batch_states = states[batch_indices]
                batch_movement_actions = movement_actions[batch_indices]
                batch_attack_actions = attack_actions[batch_indices]
                batch_old_movement_log_probs = old_movement_log_probs[batch_indices]
                batch_old_attack_log_probs = old_attack_log_probs[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                
                movement_probs, attack_probs, values = self.policy(batch_states)
                
                movement_dist = torch.distributions.Categorical(movement_probs)
                attack_dist = torch.distributions.Categorical(attack_probs)
                
                new_movement_log_probs = movement_dist.log_prob(batch_movement_actions)
                new_attack_log_probs = attack_dist.log_prob(batch_attack_actions)
                
                movement_ratio = torch.exp(new_movement_log_probs - batch_old_movement_log_probs)
                attack_ratio = torch.exp(new_attack_log_probs - batch_old_attack_log_probs)
                
                movement_surr1 = movement_ratio * batch_advantages
                movement_surr2 = torch.clamp(movement_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                attack_surr1 = attack_ratio * batch_advantages
                attack_surr2 = torch.clamp(attack_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                policy_loss = -(torch.min(movement_surr1, movement_surr2) + torch.min(attack_surr1, attack_surr2)).mean()
                
                # ensure values and returns have same shape
                values_flat = values.view(-1)
                returns_flat = batch_returns.view(-1)
                value_loss = nn.functional.mse_loss(values_flat, returns_flat)
                
                entropy = (movement_dist.entropy() + attack_dist.entropy()).mean()
                
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
        
        self.rollout_buffer = {
            'states': [],
            'movement_actions': [],
            'attack_actions': [],
            'rewards': [],
            'values': [],
            'dones': [],
            'movement_log_probs': [],
            'attack_log_probs': []
        }
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
    
    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
