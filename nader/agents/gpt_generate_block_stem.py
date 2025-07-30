import re
import json
import os
import random
import time
from datetime import datetime
import pdb
import json

from .prompts import prompt_generate_stem_downsample,prompt_develop_experience,prompt_generate_stem_downsample_nas_bench_201,prompt_generate_downsample_darts
from .gpt_generate_block_base import GPTGenerateBlockBase
from .agent_expe_retriever_dev import DevelopExperienceRetriever


class GPTGenerateBlockStemDownsample(GPTGenerateBlockBase):
    """
    Generate stem and downsample
    """

    def __init__(self, block_txts_example_dir=None, agent_name='modify_stem', dataset='imagenet-1k',mode='nas-bench',**kwargs) -> None:
        super().__init__(agent_name, **kwargs)
        self.block_txts_example_dir = block_txts_example_dir
        self.dataset = dataset
        self.mode = mode

    def get_examples(self,num=3):
        assert self.block_txts_example_dir is not None
        examples = ""
        stem_txt = None
        files = list(os.listdir(self.block_txts_example_dir))
        num = min(num,len(files))
        files = random.sample(files,num)
        for file in files:
            with open(os.path.join(self.block_txts_example_dir,file),'r') as f:
                example = f.read()
                if self.mode=='darts':
                    l = self.parse_result(example)
                    example = l[0]+'\n'+l[2]
                    stem_txt = l[1]
            examples=examples+example+'\n'
        return examples, stem_txt
    
    # def gen(self, proposal=None, block=None, example_num=3, temperature=0.1):
    #     if self.use_experience:
    #         expes = self.agent_devexpe_retriever(proposal,mode='stem',num=3)
    #         expes2 = self.agent_devexpe_retriever(proposal,mode='downsample',num=2)
    #         expes.extend(expes2)
    #         experience = '\n'.join([f"{i+3}. {expe}" for i, expe in enumerate(expes)])
    #     else:
    #         experience = ""
    #     examples = self.get_examples(num=example_num)
    #     if self.dataset=='imagenet-1k':
    #         prompt = prompt_generate_stem_downsample
    #     elif self.dataset.startswith('nas-bench-201'):
    #         prompt = prompt_generate_stem_downsample_nas_bench_201
    #     else:
    #         raise NotImplementedError
    #     prompt = prompt.format(examples=examples,experience=experience,input=block)
    #     messages = [{'role':'user','content':prompt}]
    #     response = self.call_gpt(messages,temperature)
    #     res = response['output']
    #     ret = {
    #         'output':res,
    #         'prompt_tokens':response['prompt_tokens'],
    #         'completion_tokens':response['completion_tokens'],
    #         'list':self.parse_result(res),
    #         'time':response['time']
    #     }
    #     return ret

    def run(self, proposal=None, block=None, feedback=None, example_num=3, temperature=0.1):
        if block:
            if self.use_experience:
                expes = self.agent_devexpe_retriever(proposal,mode='stem',num=3)
                expes2 = self.agent_devexpe_retriever(proposal,mode='downsample',num=2)
                expes.extend(expes2)
                experience = '\n'.join([f"{i+3}. {expe}" for i, expe in enumerate(expes)])
            else:
                experience = ""
            examples,self.stem_txt = self.get_examples(num=example_num)
            if self.mode=='darts':
                PROMPT = prompt_generate_downsample_darts
            elif self.dataset=='imagenet-1k':
                PROMPT = prompt_generate_stem_downsample
            elif 'cifar' in self.dataset.lower() or 'imagenet16-120' in self.dataset.lower():
                PROMPT = prompt_generate_stem_downsample_nas_bench_201
            else:
                raise NotImplementedError
            prompt = PROMPT.format(examples=examples,experience=experience, input=input)
            self.history = [{'role':'user','content':prompt}]
        elif feedback:
            assert len(self.history)>=1
            self.history.append({'role':'user','content':f"The block you generate has following error:{feedback},please fix it and generate a new one."})
        else:
            raise NotImplementedError
        response = self.call_gpt(self.history,temperature)
        res = response['output']
        l = self.parse_result(res)
        if self.mode=='darts':
            l = [self.stem_txt,l[0]]
            res = '\n'.join(l)
        ret = {
            'output':res,
            'prompt_tokens':response['prompt_tokens'],
            'completion_tokens':response['completion_tokens'],
            'list':l,
            'time':response['time']
        }
        return ret

