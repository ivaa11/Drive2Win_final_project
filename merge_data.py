"""Merge two or more .npz datasets into one.

Usage:
    py merge_data.py data_a.npz data_b.npz --out data_merged_new.npz
    py merge_data.py data_a.npz data_b.npz data_c.npz --out data_merged_new.npz
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("inputs", nargs="+", help="Input .npz files to merge")
ap.add_argument("--out", required=True, help="Output .npz path")
args = ap.parse_args()

all_states, all_actions, all_positions = [], [], []
seed = None

for path in args.inputs:
    d = np.load(path)
    all_states.append(d["states"])
    all_actions.append(d["actions"])
    pos = d["positions"] if "positions" in d else np.zeros((len(d["states"]), 3))
    all_positions.append(pos)
    if seed is None and "seed" in d:
        seed = d["seed"]
    print(f"  {path}: {len(d['states'])} samples")

states  = np.concatenate(all_states)
actions = np.concatenate(all_actions)
positions = np.concatenate(all_positions)

np.savez(args.out, states=states, actions=actions, positions=positions,
         seed=seed if seed is not None else 0)
print(f"Saved {args.out}: {len(states)} total samples")
