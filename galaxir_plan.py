# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 22:03:53 2019

@author: lucas_lyc
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import os
import sys
import rdkit
import random
import csv
from gretro.common.cmd_args import cmd_args
from gretro.data_process.data_info import DataInfo, load_center_maps
from gretro.api import RetroGLN
from gretro.common.evaluate import get_score, canonicalize

import json
import molvs
import random
from tqdm import tqdm
from mcts import Node, mcts
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem
starting_mols = set()
with open('data/emolecules.smi', 'r') as f:
    for line in tqdm(f, desc='Loading base compounds'):
        smi = line.strip()
        smi = molvs.standardize_smiles(smi)
        smi = Chem.MolFromSmiles(smi)
        smi = Chem.RemoveHs(smi)
        [a.ClearProp('molAtomMapNumber') for a in smi.GetAtoms()]
        smi = Chem.MolToSmiles(smi)
        starting_mols.add(smi)
print('Base compounds:', len(starting_mols))

model = '../../mydropbox/phama.ai/retrosyn_graph/model_dumps/schneider50k.ckpt'
beam_size = 10
topk = 10

def expansion(node, model):
    """Try expanding each molecule in the current state
    to possible reactants"""

    # Assume each mol is a SMILES string
    mols = node.state

    # Convert mols to format for prediction
    # If the mol is in the starting set, ignore
    mols = [mol for mol in mols if mol not in starting_mols]

    # Generate children for reactants
    children = []
    for mol in mols:
        result = model.run(mol, beam_size=50, topk=50)
        # State for children will
        # not include this mol
        new_state = mols - {mol}
        for idx in range(10):
            # Extract actual rule
            rule = result['template'][idx]

            # TODO filter_net should check if the reaction will work?
            # should do as a batch

            # Apply rule
            reactants = result['reactants'][idx]

            if not reactants: continue

            state = new_state | set(reactants)
            terminal = all(mol in starting_mols for mol in state)
            child = Node(state=state, is_terminal=terminal, parent=node, action=rule)
            children.append(child)
    return children


def rollout(node, model,  max_depth=200):
    cur = node
    for _ in range(max_depth):
        if cur.is_terminal:
            break
    
        # Select a random mol (that's not a starting mol)
        mols = [mol for mol in cur.state if mol not in starting_mols]
        mol = random.choice(mols)
        
        res = model.run(mol, beam_size, topk)
        reactants = random.choice(res['reactants']).split('.')
        
        state = cur.state | set(reactants)
        
        # State for children will
        # not include this mol
        state = state - {mol}

        terminal = all(mol in starting_mols for mol in state)
        cur = Node(state=state, is_terminal=terminal, parent=cur)

    # Max depth exceeded
    else:
        print('Rollout reached max depth')

        # Partial reward if some starting molecules are found
        reward = sum(1 for mol in cur.state if mol in starting_mols)/len(cur.state)

        # Reward of -1 if no starting molecules are found
        if reward == 0:
            return -1.

        return reward

    # Reward of 1 if solution is found
    return 1.


def plan(target_mol):
    """Generate a synthesis plan for a target molecule (in SMILES form).
    If a path is found, returns a list of (action, state) tuples.
    If a path is not found, returns None."""
    root = Node(state={target_mol})

    path = mcts(root, expansion, rollout, iterations=2000, max_depth=200)
    if path is None:
        print('No synthesis path found. Try increasing `iterations` or `max_depth`.')
    else:
        print('Path found:')
        path = [(n.action, n.state) for n in path[1:]]
    return path


if __name__ == '__main__':
    # target_mol = '[H][C@@]12OC3=C(O)C=CC4=C3[C@@]11CCN(C)[C@]([H])(C4)[C@]1([H])C=C[C@@H]2O'
    target_mol = 'CC(=O)NC1=CC=C(O)C=C1'
    root = Node(state={target_mol})
    path = plan(target_mol)
    import ipdb; ipdb.set_trace()