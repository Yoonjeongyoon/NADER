import os
import json
import random
import numpy as np
import re
import argparse

from nader.agents.agent_reader import AgentReaderFilterPapers


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def filt_papers(anno_path,abstract_dir,out_path,llm_log_dir):
    agent = AgentReaderFilterPapers(log_dir=llm_log_dir)
    with open(anno_path,'r') as f, open(out_path,'w') as out:
        lines = f.readlines()
        out.write('[\n')
        num,total = 0,len(lines)
        for i,line in enumerate(lines):
            d = json.loads(line)
            with open(os.path.join(abstract_dir,f"abstract{d['id']}.txt"),'r') as f2:
                abstract = f2.read()
            input = {'title':d['title'],'abstract':abstract}
            res,t = agent(input,temperature=0.7)
            import pdb
            pdb.set_trace()
            d['tag'] = res
            if i!=0:
                out.write(',\n')
            json.dump(d,out,indent='\t')
            out.flush()
            if (i+1)%10==0:
                print(f"{i+1}/{total},{t}s")
        out.write('\n]')


if __name__=='__main__':
    set_seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno-path')
    parser.add_argument('--abstract-dir')
    parser.add_argument('--anno-out-path')
    parser.add_argument('--llm-log-dir',default='logs/llm_response/reader-filter-paper')
    args = parser.parse_args()
    filt_papers(args.anno_path,args.abstract_dir,args.anno_out_path,args.llm_log_dir)




