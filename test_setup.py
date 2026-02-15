import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    print('testing imports...')
    
    try:
        import retro
        print('✓ retro imported successfully')
    except ImportError as e:
        print(f'✗ failed to import retro: {e}')
        return False
    
    try:
        import torch
        print(f'✓ torch imported successfully (cuda available: {torch.cuda.is_available()})')
    except ImportError as e:
        print(f'✗ failed to import torch: {e}')
        return False
    
    try:
        import cv2
        print('✓ cv2 imported successfully')
    except ImportError as e:
        print(f'✗ failed to import cv2: {e}')
        return False
    
    try:
        import numpy as np
        print('✓ numpy imported successfully')
    except ImportError as e:
        print(f'✗ failed to import numpy: {e}')
        return False
    
    return True


def test_environment():
    print('\ntesting environment creation...')
    
    try:
        import retro
        env = retro.make(game='MortalKombatII-Genesis')
        print('✓ environment created successfully')
        
        obs = env.reset()
        print(f'✓ environment reset, observation shape: {obs.shape}')
        
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f'✓ environment step, reward: {reward}')
        
        env.close()
        return True
        
    except Exception as e:
        print(f'✗ failed to create environment: {e}')
        print('  make sure you have imported the mortal kombat ii rom:')
        print('  python -m retro.import /path/to/roms/')
        return False


def test_custom_modules():
    print('\ntesting custom modules...')
    
    try:
        from preprocessing.wrappers import make_env
        env = make_env(game_name='MortalKombatII-Genesis')
        print('✓ custom wrappers work')
        
        obs = env.reset()
        print(f'✓ preprocessed observation shape: {obs.shape}')
        env.close()
        
    except Exception as e:
        print(f'✗ failed to test custom modules: {e}')
        return False
    
    try:
        from models.cnn_policy import CNNPolicy
        import torch
        
        policy = CNNPolicy(input_channels=1, num_movement_actions=5, num_attack_actions=7)
        print('✓ cnn policy created')
        
        dummy_input = torch.randn(1, 1, 84, 84)
        movement_probs, attack_probs, value = policy(dummy_input)
        print(f'✓ forward pass successful')
        print(f'  movement probs shape: {movement_probs.shape}')
        print(f'  attack probs shape: {attack_probs.shape}')
        print(f'  value shape: {value.shape}')
        
    except Exception as e:
        print(f'✗ failed to test cnn policy: {e}')
        return False
    
    try:
        from agents.ppo_agent import PPOAgent
        from models.cnn_policy import CNNPolicy
        
        policy = CNNPolicy(input_channels=1, num_movement_actions=5, num_attack_actions=7)
        agent = PPOAgent(policy, num_movement_actions=5, num_attack_actions=7)
        print('✓ ppo agent created')
        
    except Exception as e:
        print(f'✗ failed to test ppo agent: {e}')
        return False
    
    return True


def main():
    print('mortal kombat 2 rl project - setup verification\n')
    print('=' * 60)
    
    success = True
    
    success = test_imports() and success
    success = test_environment() and success
    success = test_custom_modules() and success
    
    print('\n' + '=' * 60)
    if success:
        print('all tests passed! you are ready to start training.')
        print('\nto start training, run:')
        print('  python train_ppo.py')
    else:
        print('some tests failed. please fix the issues above.')
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
