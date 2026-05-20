import numpy as np

d = np.load('data_merged.npz')
moving = d['states'][:, 0] > 1.0
print(f'keeping {moving.sum()} / {len(moving)} samples')

pos = d['positions']
if len(pos) == len(d['states']):
    pos_filtered = pos[moving]
else:
    pos_filtered = pos

np.savez('data_merged_filtered.npz',
    states=d['states'][moving],
    actions=d['actions'][moving],
    positions=pos_filtered,
    seed=d['seed'])

print('Saved data_merged_filtered.npz')
