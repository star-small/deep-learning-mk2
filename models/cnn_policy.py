import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNPolicy(nn.Module):
    def __init__(self, input_channels, num_movement_actions, num_attack_actions):
        super(CNNPolicy, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        
        self._initialize_weights()
        
        conv_output_size = self._get_conv_output_size((input_channels, 84, 84))
        
        self.fc = nn.Linear(conv_output_size, 512)
        
        self.movement_head = nn.Linear(512, num_movement_actions)
        self.attack_head = nn.Linear(512, num_attack_actions)
        
        self.value_head = nn.Linear(512, 1)
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            dummy_output = self._forward_conv(dummy_input)
            return int(np.prod(dummy_output.size()[1:]))
    
    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x
    
    def forward(self, x):
        x = x.float() / 255.0
        
        x = self._forward_conv(x)
        x = x.reshape(x.size(0), -1)
        
        x = F.relu(self.fc(x))
        
        movement_logits = self.movement_head(x)
        attack_logits = self.attack_head(x)
        
        value = self.value_head(x)
        
        movement_probs = F.softmax(movement_logits, dim=-1)
        attack_probs = F.softmax(attack_logits, dim=-1)
        
        return movement_probs, attack_probs, value
