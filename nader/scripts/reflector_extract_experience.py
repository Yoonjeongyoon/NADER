import json
import os
import re
import pdb

from agents.agent_reflect_research import ResearchReflector
from agents.agent_resexpe_retriever import ResearchExperienceRetriever


def track(log_paths,out_path):
    annos = []
    for file in log_paths:
        with open(file,'r') as f:
            ds = json.load(f)
            for key,val in ds.items():
                raw_name = val['from_block_name']
                acc = val['acc']
                if raw_name:
                    raw_acc = ds[raw_name]['acc']
                    if raw_acc*0.9 > acc:
                        anno = {
                            "raw":{
                                "name":raw_name,
                                "blocks":ds[raw_name]["blocks"],
                                "acc":raw_acc
                            },
                            "inspiration":{
                                "id":val['inspiration_id'],
                                "content":val["inspiration"]
                            },
                            "new":{
                                "name":key,
                                "blocks":val["blocks"],
                                "acc":acc
                            }
                        }
                        annos.append(anno)

    with open(out_path,'w') as f:
        json.dump(annos,f,indent='\t')
    print(f"Log number:{len(annos)}")

def reflect(log_path,out_path):
  
    def flush_write(anno,path):
        with open(path,'w') as f:
            json.dump(anno,f,indent='\t')
    agent = ResearchReflector(log_dir='logs/extract_experience_research')
    if os.path.isfile(out_path):
        with open(out_path,'r') as f:
            annos = json.load(f)
    annos = []
    with open(log_path,'r') as f:
        ds = json.load(f)
        for i,d in enumerate(ds):
            print(f"{i}/{len(ds)}")
            exp = agent(d)['output']
            if isinstance(exp,str):
                d['experience'] = exp
                annos.append(d)
                flush_write(annos,out_path)
            else:
                print(f"error:{i}")

def reoragnize(raw_path,new_path):
    with open(raw_path,'r') as f:
        ds = json.load(f)
    res = {}
    for d in ds:
        if d['inspiration']['content'] not in res:
            res[d['inspiration']['content']] = []
        res[d['inspiration']['content']].append(d['experience'])
    with open(new_path,'w') as f:
        json.dump(res,f,indent='\t')


def construct_db(table_name,exp_path):
    agent = ResearchExperienceRetriever()
    agent.create_table(table_name,exp_path)

def test_db(table_name):
     agent = ResearchExperienceRetriever(table_name)
     exprs = agent('Integrate spatial attention mechanisms to enhance the focus on relevant spatial features, potentially improving the accuracy on tasks where spatial relationships are crucial.')
     pdb.set_trace()


if __name__=='__main__':
    log_paths = [
        'logs/nas-bench-201/cifar10/resnet_basic_random_greedy-width10/mog.json',
        'logs/nas-bench-201/cifar100/resnet_basic_random_greedy-width10/mog.json',
        'logs/nas-bench-201/imagenet16-120/resnet_basic_random_greedy-width10/mog.json',
        'logs/nader_trail3_convnext_random_greedy_3393/mog.json',
        'logs/nader_trail2_convnext_random_greedy/mog.json',
        'logs/nader_trail1_convnext_reflection/mog.json',
        'logs/nader_trail1_resnet_bottle_reflection/mog.json',
        'logs/nader_trail1_convnext_random/mog.json'
    ]
    tag = 'fail_240807'
    log_path = f'database/experiences/research_{tag}_log.json'
    # track(log_paths,log_path)

    reflectout_path = f'database/experiences/research_{tag}_reflectout.json'
    # reflect(log_path,reflectout_path)

    experience_path = f'database/experiences/research_{tag}_experience.json'
    # reoragnize(reflectout_path,experience_path)


    table_name = f'research_{tag}_experience'
    # construct_db(table_name,experience_path)

    test_db(table_name)