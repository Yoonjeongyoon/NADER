import re
import json
import openai
from openai import OpenAI
import os
import random
import time
from datetime import datetime
import pdb
import requests
import json

from .prompts import prompt_reflector_develop_allfailed_base, prompt_reflector_develop_allfailed_stem, prompt_reflector_develop_allfailed_downsample, prompt_reflector_develop_allfailed_all
from .base_agent import BaseAgent

class DevelopRelector(BaseAgent):


    def __init__(self, model_name='develop_reflector', log_dir='logs') -> None:
        super().__init__(model_name, log_dir)
        self.mode2prompt = {
            'base':prompt_reflector_develop_allfailed_base,
            'stem':prompt_reflector_develop_allfailed_stem,
            'downsample':prompt_reflector_develop_allfailed_downsample,
            'all':prompt_reflector_develop_allfailed_all
        }

        

    def __call__(self, block, error, mode='base', temperature=0.7):
        """
        proposal:{"name":...,"operation":...}
        blocks:str
        """
        prompt = self.mode2prompt[mode].format(block=block,error=error)
        messages = [{'role':'user','content':prompt}]
        response = super().__call__(messages,temperature=temperature)
        if self.log_dir:
            TIME_NOW = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            save_path = os.path.join(self.log_dir,TIME_NOW+'.json')
            with open(save_path,'w') as f:
                json.dump(response,f,indent='\t')
        res = response['output']
        ret = {
            'output':self.parse_result(res),
            'prompt_tokens':response['prompt_tokens'],
            'completion_tokens':response['completion_tokens'],
            'time':response['time']
        }
        return ret

    def parse_result(self,txt):
        s = re.findall('<tip>(.*)</tip>',txt,re.DOTALL)
        if len(s)>0:
            return s[0].strip()
        return None
    


