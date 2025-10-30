import os
import json
import numpy as np
import copy

class ProposalManagement:

    def __init__(self,database_proposal_dir) -> None:
        self.database_proposal_dir = database_proposal_dir
        os.makedirs(database_proposal_dir,exist_ok=True)
        self.anno_path = os.path.join(database_proposal_dir,'annos.json')
        self.load_annos(self.anno_path)
        self.load_proposals()
    
    def __getitem__(self,i):
        return list(self.anno.keys())[i]
    
    def __contains__(self,name):
        return name in self.anno
    
    def __len__(self):
        return len(self.anno)

    def load_annos(self,path):
        if os.path.isfile(path):
            with open(path,'r') as f:
                ds = json.load(f)
        else:
            ds = {}
        self.anno = ds
        return ds
    
    def load_proposals(self):
        if 'mutate' in self.anno:
            p1 = list(self.anno['mutate'])
        else:
            p1 = []
        if 'cross' in self.anno:
            p2 = list(self.anno['cross'])
        else:
            p2 = []
        p3 = copy.deepcopy(p1)
        p3.extend(copy.deepcopy(p2))
        self.proposals = {}
        self.proposals['all'] = p3
        self.proposals['mutate'] = p1
        self.proposals['cross'] = p2
        return self.proposals
    
    def sample(self,num,type='all',weighted=False,ret='name'):
        if weighted:
            raise NotImplementedError
        else:
            proposals = np.random.choice(self.proposals[type],num)
        if ret=='name':
            return proposals
        elif ret=='name-operation':
            res = []
            for proposal in proposals:
                res.append({
                    'name':proposal,
                    'operation':self.anno[self.get_type(proposal)][proposal]['operation']
                })
            return res
        elif ret=='name-reason-operation':
            res = []
            for proposal in proposals:
                res.append({
                    'name':proposal,
                    'reason':self.anno[self.get_type(proposal)][proposal]['reason'],
                    'operation':self.anno[self.get_type(proposal)][proposal]['operation']
                })
            return res
    
    def get_type(self,name):
        if name in self.proposals['mutate']:
            return 'mutate'
        elif name in self.proposals['cross']:
            return 'cross'
        else:
            raise NotImplementedError