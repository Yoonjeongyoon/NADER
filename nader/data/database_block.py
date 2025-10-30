import os
import json
import re
import numpy as np
from prettytable import PrettyTable

class BlockManagement:

    def __init__(self,database_block_dir='database/blocks',
                 train_dir='logs/models',
                 model_factory_dir='ModelFactory') -> None:
        self.train_dir = train_dir
        self.model_factory_dir = model_factory_dir
        self.database_block_dir = database_block_dir
        os.makedirs(database_block_dir,exist_ok=True)
        self.anno_path = os.path.join(database_block_dir,'annos.json')
        self.anno = self.load_annos(self.anno_path)

    def __getitem__(self,i):
        return list(self.anno.keys())[i]
    
    def __contains__(self,name):
        return name in self.anno
    
    def __len__(self):
        return len(self.anno)

    def get_acc(self,name):
        self.anno = self.load_annos(self.anno_path)
        assert name in self.anno
        return self.anno[name]['test']['accuracy']

    def load_annos(self,path):
        if os.path.isfile(path):
            with open(path,'r') as f:
                ds = json.load(f)
        else:
            ds = {}
        return ds

    def get_block_base_txt(self,block_names):
        if isinstance(block_names,list):
            txts = []
            for block_name in block_names:
                base_name = self.anno[block_name]['blocks']['base']
                path = os.path.join(self.model_factory_dir,'blocks','base','txt',base_name+'.txt')
                assert os.path.isfile(path)
                with open(path,'r') as f:
                    txt = f.read()
                    txts.append(txt)
        else:
            base_name = self.anno[block_names]['blocks']['base']
            path = os.path.join(self.model_factory_dir,'blocks','base','txt',base_name+'.txt')
            assert os.path.isfile(path),path
            with open(path,'r') as f:
                txts = f.read()
        return txts

    
    def sample(self,num,weighted=False,ret='name'):
        if weighted:
            raise NotImplementedError
        else:
            blocks = np.random.choice(self,num)
        if ret=='name':
            return blocks
        elif ret=='base':
            bases = []
            for block in blocks:
                bases.append()

    def update_block(self,block_name,froms):
        res = {}
        with open(os.path.join(self.model_factory_dir,'models','anno.json'),'r') as f:
            ds = json.load(f)
            model_name = block_name.replace('block','model')
            if model_name not in ds:
                return {'error':"Can't find model in ModelFactory."}
            res = ds[model_name]
        res_path = os.path.join(self.train_dir,model_name,'results.json')
        if not os.path.isfile(res_path):
            return {'error':"Training has not done."}
        with open(res_path,'r') as f:
            ds = json.load(f)
            if 'error' in ds:
                return ds
            if 'test' not in ds:
                return {'error':"Training has not done."}
            res['test'] = ds['test']
        if os.path.isfile(self.anno_path):
            with open(self.anno_path,'r') as f:
                ds = json.load(f)
            if block_name in ds:
                if res!=ds[block_name]:
                    return {'error':'Dupliate error.'}
                print(f'Block {block_name} has existed in database.')
                return True
        else:
            ds = {}
        res['from'] = froms
        ds[block_name] = res
        with open(self.anno_path,'w') as f:
            json.dump(ds,f,indent='\t')
        self.anno = self.load_annos(self.anno_path)
        return True
    
    def update_blocks(self,block_names=[],froms={}):
        success = []
        errors = {}
        for block_name in block_names:
            res = self.update_block(block_name,froms[block_name])
            if res!=True and 'error' in res:
                errors[block_name] = res['error']
            else:
                success.append(block_name)
        return success, errors
    
    def print_table(self):
        with open(self.anno_path,'r') as f:
            ds = json.load(f)
        table = PrettyTable()
        names = ['block_name','params','accuracy']
        table.field_names = names
        table.align = 'l'
        for model_name in ds:
            table.add_row([model_name,ds[model_name]['params'],ds[model_name]['test']['accuracy']])
        print(table)

    @staticmethod
    def txt2json(txt):
        s = re.findall('##[^#]*##[^#]*',txt,re.DOTALL)
        res = {
            'base':s[0],
            'stem':s[1],
            'downsample':s[2]
        }
        return res
        

                    
                
