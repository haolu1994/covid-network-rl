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
import csv
from gretro.common.cmd_args import cmd_args
from gretro.data_process.data_info import DataInfo, load_center_maps
from gretro.api import RetroGLN
from gretro.common.evaluate import get_score, canonicalize

import torch
import json
import molvs
import random
from tqdm import tqdm
from mcts import Node, mcts
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem

import argparse
cmd_opt = argparse.ArgumentParser(description='Argparser for test only')
cmd_opt.add_argument('-model_for_test', default=None, help='model for test')
cmd_opt.add_argument('-target_mol', default=None, help='mol for test')
local_args, _ = cmd_opt.parse_known_args()

starting_mols = set()

if os.path.exists('data/standardized_smi.txt'):
    with open('data/standardized_smi.txt', 'r') as f:
        for line in tqdm(f, desc='Loading standardized compounds'):
            smi = line.strip()
            starting_mols.add(smi)
else:
    f2 = open('data/standardized_smi.txt', "w")
    with open('data/emolecules.smi', 'r') as f:
        for line in tqdm(f, desc='Loading base compounds'):
            smi = line.strip()
            try:
                smi = molvs.standardize_smiles(smi)
            except:
                smi = smi
            # try:
            #     smi = Chem.MolFromSmiles(smi)
            #     smi = Chem.RemoveHs(smi)
            #     [a.ClearProp('molAtomMapNumber') for a in smi.GetAtoms()]
            #     smi = Chem.MolToSmiles(smi)
            #     starting_mols.add(smi)
            #     print(smi)
            # except:
            #     continue
            f2.write(smi + '\n')
    print('Base compounds:', len(starting_mols))



def expansion(node):
    """Try expanding each molecule in the current state
    to possible reactants"""

    # Assume each mol is a SMILES string
    mols = node.state

    # Convert mols to format for prediction
    # If the mol is in the starting set, ignore
    mols = [mol for mol in mols if mol not in starting_mols]
    mols = set(mols)

    # Generate children for reactants
    children = []
    for mol in mols:
        result = model.run(mol, beam_size=10, topk=10)
        # State for children will
        # not include this mol
        new_state = mols - {mol}
        if result != None:
            reactants = result['reactants']
            template = result['template']
            if not reactants: continue
            for i in range(len(reactants)):
                state = new_state | set(reactants[i].split('.'))
                rule = template[i]
                terminal = all(mol in starting_mols for mol in state)
                child = Node(state=state, is_terminal=terminal, parent=node, action = rule)
                children.append(child)
    return children


def rollout(node, max_depth=20):
    cur = node
    for _ in range(max_depth):
        if cur.is_terminal:
            break

        # Select a random mol (that's not a starting mol)
        mols = [mol for mol in cur.state if mol not in starting_mols]
        mol = random.choice(mols)
        try: 
            res = model.run(mol, beam_size=10, topk=10)
            index= random.choice(range(len(res['reactants'])))
            reactants = res['reactants'][index]
            rule = res['template'][index]
            reactants = random.choice(res['reactants']).split('.')
        except:
            reactants = {}
            rule = {}
        #try:
        #    reactants = {molvs.standardize_smiles(smi) for smi in reactants}   
        #except:
        #    reactants = reactants
        state = cur.state | set(reactants)

        # State for children will
        # not include this mol
        state = state - {mol}
        terminal = all(mol in starting_mols for mol in state)
        cur = Node(state=state, is_terminal=terminal, parent=cur, action = rule)

    # Max depth exceeded
    else:
        print('Rollout reached max depth')
        # Partial reward if some starting molecules are found
        reward = sum(1 for mol in cur.state if mol in starting_mols) / len(cur.state)
        print('rollout', reward)
        # Reward of -1 if no starting molecules are found
        if reward == 0:
            return -1.

        return reward

    # Reward of 1 if solution is found
    return 1.


def plan(target_mol, retro_path):
    """Generate a synthesis plan for a target molecule (in SMILES form).
    retro_path is a file path to store the multistep retrosynthesis path
    If a path is found, returns a list of (action, state) tuples.
    If a path is not found, returns None."""
    root = Node(state={target_mol})

    path = mcts(root, expansion, rollout, iterations= 1000, max_depth=200)
    if path is None:
        print('No synthesis path found. Try increasing `iterations` or `max_depth`.')
    else:
        print('Path found:')
        f = open(retro_path , 'a')
        for n in path[1:]:
            f.write(str(n.action)+ '\n')
        f.close()
        path = [(n.action, n.state) for n in path]
    return path


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    # model_path = '../../mydropbox/phama.ai/retrosyn_graph/model_dumps/schneider50k.ckpt'
    # cooked_root ='../cooked_data'
    model = RetroGLN(cmd_args.cooked_root, local_args.model_for_test)
    target_mol = local_args.target_mol
    # beam_size = 10
    # topk = 10
    # target_mol = '[H][C@@]12OC3=C(O)C=CC4=C3[C@@]11CCN(C)[C@]([H])(C4)[C@]1([H])C=C[C@@H]2O'
    # target_mol = 'CC(=O)NC1=CC=C(O)C=C1'
    # target_mols = ['C1=CC(=C(C(=C1)F)CN2C=C(N=N2)C(=O)N)F', 'NCCN[S](=O)(=O)c1ccc(NC(=S)NC2CCCC2)cc1', 'CN1CCN(CC1)C2=C(C=C(C=C2)C3=CC=CC(=C3)CN4CCOCC4)NC(=O)C5=CN=C(O)C=C5C(F)(F)F']  
    # for target_mol in target_mols:
    root = Node(state={target_mol})
    path = plan(target_mol, 'Retro_Path.txt')
    print(path)
    #import ipdb;

    #ipdb.set_trace()
