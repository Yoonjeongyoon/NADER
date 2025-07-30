import json
from typing import *
import pdb
import re

from .base_agent import BaseAgent
from .prompts import PROMPT_EVAL_DEVELOP


class EvaluatorDevelop(BaseAgent):


    def __init__(self,*args,**kwargs):
        super().__init__(agent_name='eval_develop',*args,**kwargs)

    def __call__(self,raw,insp,pred,ref,**kwargs):
        prompt = PROMPT_EVAL_DEVELOP.format(raw_block=raw,inspiration=insp,pred_block=pred,ref_block=ref)
        message = [{'role':'user','content':prompt}]
        res = super().__call__(message,**kwargs)
        m = re.findall('<answer>(.*?)</answer>',res['output'],re.DOTALL)
        assert len(m)>0
        m = m[0]
        if 'yes' in m.lower():
            return True
        elif 'no' in m.lower():
            return False
        else:
            not NotImplementedError
        return res
