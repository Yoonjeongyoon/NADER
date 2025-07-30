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

from .prompts import prompt_reflector_research_failed
from .base_agent import BaseAgent

class ResearchReflector(BaseAgent):


    def __init__(self, model_name='research_reflector', log_dir='logs') -> None:
        super().__init__(model_name, log_dir)
        

    def __call__(self, anno, temperature=0.7):
        prompt = prompt_reflector_research_failed.format(raw=anno['raw']['blocks'][0],raw_acc=anno['raw']['acc'],new=anno['new']['blocks'][0],new_acc=anno['new']['acc'])
        messages = [{'role':'user','content':prompt}]
        response = super().__call__(messages,temperature=temperature)
        response['output'] = self.parse_result(response['output'])
        return response

    def parse_result(self,txt):
        s = re.findall('<suggestion>(.*)</suggestion>',txt,re.DOTALL)
        if len(s)>0:
            return s[0].strip()
        return None
    


