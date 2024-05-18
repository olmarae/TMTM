import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime as dt
import json

print('loading raw data')
with open('processed_data/uid_index.json', 'r') as f:
    uid_index = json.load(f)

edge=pd.read_csv("Dataset/edge.csv")

print('extracting edge_index&edge_type')
edge_index=[]
edge_type=[]
for i in tqdm(range(len(edge))):
    sid=edge['source_id'][i]
    tid=edge['target_id'][i]
    if edge['relation'][i]=='followers':
        try:
            edge_index.append([uid_index[sid],uid_index[tid]])
            edge_type.append(0)
        except KeyError:
            continue
    elif edge['relation'][i]=='following':
        try:
            edge_index.append([uid_index[sid],uid_index[tid]])
            edge_type.append(1)
        except KeyError:
            continue

list_membership = {}

for i in range(len(edge)):
    sid = edge['source_id'][i]
    tid = edge['target_id'][i]
    relation = edge['relation'][i]
    
    if relation == 'followers':
        try:
            edge_index.append([uid_index[sid],uid_index[tid]])
            edge_type.append(0)
        except KeyError:
            continue
    elif relation=='following':
        try:
            edge_index.append([uid_index[sid],uid_index[tid]])
            edge_type.append(1)
        except KeyError:
            continue
    elif relation == 'own':
        if tid not in list_membership:
            list_membership[tid] = {'creator': [], 'members': []}
        list_membership[tid]['creator'].append(uid_index[sid])
    elif relation in ['followed', 'membership']:
        if sid not in list_membership:
            list_membership[sid] = {'creator': [], 'members': []}
        list_membership[sid]['members'].append(uid_index[tid])       

# Create relation 2: Ownership
for list_id, roles in list_membership.items():
    # Conexiones del creador a cada miembro
    for creator in roles['creator']:
        for member in roles['members']:
            try:
                edge_index.append([creator, member])  # Creador a miembro
                edge_type.append(2)  # Ownership
            except KeyError:
                continue

torch.save(torch.LongTensor(edge_index).t(),"./processed_data/edge_index.pt")
torch.save(torch.LongTensor(edge_type),"./processed_data/edge_type.pt")