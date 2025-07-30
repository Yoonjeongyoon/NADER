import json
import os
import argparse

def merge_inspirations(root_dir,out_path):
    res = []
    num = 0
    for file in os.listdir(root_dir):
        with open(os.path.join(root_dir,file),'r') as f:
            ds = json.load(f)
            if ds['proposals'] is None:
                continue
            for proposal in ds['proposals']:
                num+=1
                res.append({
                    'id':num,
                    'paper_id':ds['id'],
                    'proposal':proposal
                })
    with open(out_path,'w') as f:
        json.dump(res,f,indent='\t')

    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspirations-dir', default='data/papers/cvpr-2023/txts_inspirations')
    parser.add_argument('--out-path', default='data/papers/cvpr-2023/inspirations.json')
    args = parser.parse_args()
    merge_inspirations(args.inspirations_dir,args.out_path)
    


