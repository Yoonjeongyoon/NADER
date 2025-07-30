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

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from .prompts import prompt_reflector_develop_allfailed_base, prompt_reflector_develop_allfailed_stem, prompt_reflector_develop_allfailed_downsample, prompt_reflector_develop_allfailed_all



class DevelopExperienceRetriever():


    def __init__(self, table_name='develop_allfailed_240709_experience', model_name='develop_expr_retriever',
            db_dir='database/ChromDB/develop_experience') -> None:
        self.db_dir = db_dir
        self.embedding_func = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            max_retries=10)
        self.db  = Chroma(collection_name=table_name,embedding_function=self.embedding_func,persist_directory=db_dir)

    def create_table(self,table_name,expr_path):
        db = Chroma(collection_name=table_name,embedding_function=self.embedding_func,persist_directory=self.db_dir)
        txts,metas = [],[]
        with open(expr_path,'r') as f:
            ds = json.load(f)
            for k,d in ds.items():
                txts.append(d['inspiration'])
                metas.append({'data':json.dumps(d)})
        db.add_texts(texts=txts,metadatas=metas)
            

    def __call__(self, query, mode='base', num=5):
        """
        proposal:{"name":...,"operation":...}
        blocks:str
        """
        docs = self.db.similarity_search(query,k=10)
        rets,errors = [],[]
        for doc in docs:
            ds = json.loads(doc.metadata['data'])[mode]
            for d in ds:
                if d['error'] not in errors:
                    errors.append(d['error'])
                    rets.append(d['experience']) 
        if len(rets)>num:
            rets=rets[:5]
        return rets
    


class DevelopExperienceRetrieverRandom():


    def __init__(self, expe_path='database/experiences/develop_allfailed_240709_experience.json') -> None:
        self.expe_path = expe_path

    def __call__(self, query, mode='base', num=5):
        with open(self.expe_path,'r') as f:
            ds = json.load(f)
            exps = []
            for d in ds:
                for exp in d[mode]:
                    exps.append(exp['experience'])
        total = len(exps)
        while total<num:
            exps = exps + exps
            total*=2
        ids = np.random.choice(list(range(total)),num,replace=False)
        return np.array(exps)[ids]