import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from block_factory import BlockFactory, DAGError
import json
import os
import re
import pdb

class BlockGen:

    def __init__(self,blocks_dir='./blocks',register_path='ModelFactory.register',stem_down_scale=1,mode='nas-bench') -> None:
        self.base_block = BlockFactory(os.path.join(blocks_dir,'base'),type='base',register_path=register_path,mode=mode)
        self.stem_block = BlockFactory(os.path.join(blocks_dir,'stem'),type='stem',register_path=register_path,stem_down_scale=stem_down_scale,mode=mode)
        self.downsample_block = BlockFactory(os.path.join(blocks_dir,'downsample'),type='downsample',register_path=register_path,mode=mode)
        self.anno_path = os.path.join(blocks_dir,'anno_pairs.json')
        self.annos = self.load_annos()

    def load_annos(self):
        if os.path.isfile(self.anno_path):
            with open(self.anno_path,'r') as f:
                ds = json.load(f)
        else:
            ds = {}
        return ds

    def add_pair(self,name,base,stem,downsample):
        self.annos[name] = {
            'base':base,
            'stem':stem,
            'downsample':downsample
        }
        with open(self.anno_path,'w') as f:
            json.dump(self.annos,f,indent='\t')
    
    def delete_pair(self,name):
        del self.annos[name]
        with open(self.anno_path,'w') as f:
            json.dump(self.annos,f,indent='\t')


    def add_blocks_from_txt_path(self, path, with_isomorphic=True):
        """
        -1: existed
        {'error':...} error
        True created
        """
        id,_ = os.path.splitext(os.path.basename(path))
        if id in self.annos:
            return {'error':f'{id} existed'}
        with open(path,'r') as f:
            blocks = f.read()
        pattern = '(##(.*?)##(.(?!##))*)'
        matches = re.findall(pattern, blocks, flags=re.MULTILINE|re.DOTALL)
        
        # Handle detection mode where only base block is present
        if self.base_block.mode == 'detection':
            assert len(matches) >= 1, f"Detection mode requires at least 1 block, got {len(matches)}"
            # For detection mode, treat all sections as base blocks
            name2s = {}
            for i, match in enumerate(matches):
                name = match[1]
                s = match[0].strip('\n')
                name2s[f'base_{i}'] = s
        else:
            assert len(matches)==3, f"Expected 3 blocks (base, stem, downsample), got {len(matches)}"
            name2s = {}
            for i,match in enumerate(matches):
                name = match[1]
                s = match[0].strip('\n')
                if 'stem' in name or i==1:
                    name2s['stem'] = s
                elif 'downsample' in name or i==2:
                    name2s['downsample'] = s
                else:
                    name2s['base'] = s
        # Handle detection mode
        if self.base_block.mode == 'detection':
            # For detection mode, process all sections as base blocks
            base_ids = []
            for key, block_txt in name2s.items():
                out = self.base_block.check(block_txt, with_isomorphic=with_isomorphic)
                if isinstance(out, dict):
                    return out
                elif out == -1:
                    base_id = self.base_block.add_block(block_txt, f"{id}_{key}")
                    if isinstance(base_id, dict):
                        return base_id
                    base_ids.append(base_id)
                else:
                    base_ids.append(out)
            return True
        elif set(name2s.keys()) == set(['base','stem','downsample']):
            out = self.base_block.check(name2s['base'],with_isomorphic=with_isomorphic)
            if isinstance(out,dict):
                return out
            elif out==-1:
                out2 = self.stem_block.check(name2s['stem'],with_isomorphic=with_isomorphic)
                if isinstance(out2,dict):
                    return out2
                out3 = self.downsample_block.check(name2s['downsample'],with_isomorphic=with_isomorphic)
                if isinstance(out3,dict):
                    return out3
                base_id = self.base_block.add_block(name2s['base'],id)
                if isinstance(base_id,dict):
                    return base_id
                if out2==-1:
                    stem_id = self.stem_block.add_block(name2s['stem'],id)
                    if isinstance(stem_id,dict):
                        return stem_id
                else:
                    stem_id=out2
                if out3==-1:
                    downsample_id = self.downsample_block.add_block(name2s['downsample'],id)
                    if isinstance(downsample_id,dict):
                        return downsample_id
                else:
                    downsample_id=out3
            else:
                base_id = self.base_block.add_block(name2s['base'],id)
                annos = self.load_annos()
                if out.endswith('_base') and not out.endswith('resnet_base'):
                    out = out[:-5]
                stem_id = annos[out]['stem']
                downsample_id = annos[out]['downsample']
            self.add_pair(id,base_id,stem_id,downsample_id)
        else:
            raise NotImplementedError
        return True

    def delete_blocks(self,ids):
        for id in ids:
            self.delete_pair(id)
            self.base_block.delete_block(id)
    
    def add_blocks_from_txt_dir(self,txt_dir):
        for file in os.listdir(txt_dir):
            path = os.path.join(txt_dir,file)
            if os.path.isfile(path):
                print(path)
                status = self.add_blocks_from_txt_path(path)
                if status!=True:
                    print(status)

if __name__=='__main__':
    gen = BlockGen(blocks_dir='./blocks')
    # block이 잘 되는지 테스트용 같음 
    gen.add_blocks_from_txt_path('./block.txt')

        