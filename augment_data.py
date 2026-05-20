import numpy as np

d = np.load('data_merged.npz')
states  = d['states']
actions = d['actions']

# Flip left-right: negate heading_error, swap symmetric rays, negate steering
flipped_states  = states.copy()
flipped_actions = actions.copy()

flipped_states[:, 1]  = -states[:, 1]       # heading_error
flipped_states[:, 4]  =  states[:, 10]      # ray +45  <-> ray -45
flipped_states[:, 10] =  states[:, 4]
flipped_states[:, 5]  =  states[:, 9]       # ray +90  <-> ray -90
flipped_states[:, 9]  =  states[:, 5]
flipped_states[:, 6]  =  states[:, 8]       # ray +135 <-> ray -135
flipped_states[:, 8]  =  states[:, 6]
flipped_actions[:, 1] = -actions[:, 1]      # steering

aug_states  = np.concatenate([states,  flipped_states])
aug_actions = np.concatenate([actions, flipped_actions])
aug_pos     = np.concatenate([d['positions'], d['positions']])

np.savez('data_merged_augmented.npz',
    states=aug_states, actions=aug_actions, positions=aug_pos, seed=d['seed'])
print(f'Original: {len(states)}  Augmented: {len(aug_states)}')
