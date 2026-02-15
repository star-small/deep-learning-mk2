import numpy as np


movement_buttons = [
    [],
    ['LEFT'],
    ['RIGHT'],
    ['UP'],
    ['DOWN'],
]

attack_buttons = [
    [],
    ['A'],
    ['B'],
    ['C'],
    ['X'],
    ['Y'],
    ['Z'],
]


def create_action_from_indices(movement_idx, attack_idx):
    buttons = []
    if movement_idx < len(movement_buttons):
        buttons.extend(movement_buttons[movement_idx])
    if attack_idx < len(attack_buttons):
        buttons.extend(attack_buttons[attack_idx])
    return buttons


def buttons_to_action_array(buttons, button_list):
    action = np.zeros(len(button_list), dtype=np.int8)
    for button in buttons:
        if button in button_list:
            action[button_list.index(button)] = 1
    return action


def get_retro_button_list():
    return ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
