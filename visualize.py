import matplotlib.pyplot as plt
import numpy as np
import os
import glob


def plot_training_progress(log_file):
    episodes = []
    rewards = []
    avg_rewards = []
    
    with open(log_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                episodes.append(int(parts[0]))
                rewards.append(float(parts[1]))
                avg_rewards.append(float(parts[2]))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(episodes, rewards, alpha=0.3, label='episode reward')
    ax1.plot(episodes, avg_rewards, linewidth=2, label='average reward')
    ax1.set_xlabel('episode')
    ax1.set_ylabel('reward')
    ax1.set_title('training progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    window_size = 50
    if len(rewards) >= window_size:
        smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(episodes[:len(smoothed)], smoothed, linewidth=2)
        ax2.set_xlabel('episode')
        ax2.set_ylabel(f'reward (smoothed, window={window_size})')
        ax2.set_title('smoothed training progress')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = log_file.replace('.txt', '_plot.png')
    plt.savefig(output_file, dpi=150)
    print(f'plot saved to {output_file}')
    
    plt.show()


def plot_comparison(log_files, labels):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for log_file, label in zip(log_files, labels):
        episodes = []
        avg_rewards = []
        
        with open(log_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    episodes.append(int(parts[0]))
                    avg_rewards.append(float(parts[2]))
        
        ax.plot(episodes, avg_rewards, linewidth=2, label=label)
    
    ax.set_xlabel('episode')
    ax.set_ylabel('average reward')
    ax.set_title('training comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logs/comparison_plot.png', dpi=150)
    print('comparison plot saved to logs/comparison_plot.png')
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, help='path to log file')
    parser.add_argument('--compare', action='store_true', help='compare multiple runs')
    
    args = parser.parse_args()
    
    if args.compare:
        log_files = sorted(glob.glob('logs/training_*.txt'))
        if len(log_files) > 0:
            labels = [os.path.basename(f).replace('training_', '').replace('.txt', '') for f in log_files]
            plot_comparison(log_files, labels)
        else:
            print('no log files found in logs/ directory')
    elif args.log_file:
        plot_training_progress(args.log_file)
    else:
        log_files = sorted(glob.glob('logs/training_*.txt'))
        if len(log_files) > 0:
            latest_log = log_files[-1]
            print(f'plotting latest log file: {latest_log}')
            plot_training_progress(latest_log)
        else:
            print('no log files found. specify --log_file or run training first.')
