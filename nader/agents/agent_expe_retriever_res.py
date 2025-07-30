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
import numpy as np

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma



class ResearchExperienceRetriever():


    def __init__(self, table_name='research_experience', model_name='research_expr_retriever',
            db_dir='database/ChromDB/research_experience') -> None:
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
                txts.append(k)
                metas.append({'data':json.dumps({"exps":d})})
        db.add_texts(texts=txts,metadatas=metas)
            

    def __call__(self, query, num=5):
        docs = self.db.similarity_search(query,k=10)
        rets,errors = [],[]
        for doc in docs:
            ds = json.loads(doc.metadata['data'])['exps']
            rets.append(np.random.choice(ds,1).tolist())
        if len(rets)>num:
            rets=rets[:num]
        return rets
    

class ResearchExperienceRetrieverRandom():

    def __init__(self, expe_path='database/experiences/research_fail_240807_experience.json') -> None:
        self.expe_path = expe_path

    def __call__(self, query, num=5):
        with open(self.expe_path,'r') as f:
            ds = json.load(f)
            exps = []
            for d in ds:
                exps.extend(exp)
        total = len(exps)
        while total<num:
            exps = exps + exps
            total*=2
        ids = np.random.choice(list(range(total)),num,replace=False)
        return np.array(exps)[ids]
