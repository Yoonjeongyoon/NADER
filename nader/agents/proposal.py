import random
import numpy as np
import json
import os


class ProposalSampler():

    def __init__(self,proposals_path,log_dir='logs'):
        self.proposals_path = proposals_path
        self.proposal_label_path = os.path.join(log_dir,'proposals_label.json')
        with open(proposals_path,'r') as f:
            ds = json.load(f)
        if not os.path.isfile(self.proposal_label_path):
            for d in ds:
                d['users']=[]
            with open(self.proposal_label_path,'w') as f:
                json.dump(ds,f,indent='\t')
        self.total_num = len(ds)

    def get_proposals(self,inds):
        proposals = []
        with open(self.proposals_path,'r') as f:
            ds = json.load(f)
            for ind in inds:
                proposals.append(ds[ind]['proposal'])
        return proposals
    
    def random_sample(self,blocks,num):
        assert isinstance(blocks,list),blocks
        candicated_list = []
        with open(self.proposal_label_path,'r') as f:
            ds =json.load(f)
            for i,d in enumerate(ds):
                if len(set(blocks)&set(d['users']))==0:
                    candicated_list.append(i)
        assert len(candicated_list)>=num
        inds = random.sample(candicated_list,num)
        return np.array(self.get_proposals(inds)), np.array(inds)+1
    
    def set_used(self,name,id):
        with open(self.proposal_label_path,'r') as f:
            ds = json.load(f)
        ds[id-1]['users'].append(name)
        with open(self.proposal_label_path,'w') as f:
            json.dump(ds,f,indent='\t')

    
        
