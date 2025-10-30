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
import numpy as np

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from .prompts import PROMPT_RESEARCH_REFLECTION_INSPIRATION
from .base_agent import BaseAgent


#임베딩 기반 유사도 검색형 인스피레이션 검색기 
class InspirationRetriever():

#컬렉션(=테이블) 생성/접속. persist_directory로 로컬에 인덱스 유지.
    def __init__(self, 
            table_name='inspirations_040611',
            db_dir='database/ChromDB/inspirations') -> None:
        self.db_dir = db_dir
        self.embedding_func = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            max_retries=10)
        self.db  = Chroma(collection_name=table_name,embedding_function=self.embedding_func,persist_directory=db_dir)
#db내 문서 수 
    def __len__(self):
        return self.db._collection.count()

    def create_table(self,insp_path):
        txts,metas,ids = [],[],[]
        with open(insp_path,'r') as f:
            ds = json.load(f)
            for d in ds:
                txts.append(d['inspiration'])
                metas.append({'data':json.dumps(d),'used':False})
                ids.append(str(d['id']))
        self.db.add_texts(texts=txts,metadatas=metas,ids=ids)

    def __getitem__(self,id):
        if isinstance(id,int):
            id = str(id)
        return self.db.get([id])['documents'][0]
            
#
    def __call__(self, query, num=10, reverse=False):
        if not reverse:
            docs = self.db.similarity_search(query,k=num,filter={'used':False})
        else:
            docs = self.db.similarity_search(query,k=len(self)-int(len(self)/10),filter={'used':False})
            docs = docs[-num-1:-1]
        rets = {}
        for doc in docs:
            ds = json.loads(doc.metadata['data'])
            rets[str(ds['id'])] = ds['inspiration']
        return rets

    def set_used(self,insp_ids):
        if not isinstance(insp_ids,list):
            insp_ids = [insp_ids]
        insp_ids = [str(id) for id in insp_ids]
        res = self.db.get(insp_ids)
        for m in res['metadatas']:
            m['used']=True
        docs = [Document(page_content=content,metadata=metadata) for content,metadata in zip(res['documents'],res['metadatas'])]    
        self.db.update_documents(ids=insp_ids,documents=docs)


class InspirationSamplerReflection(BaseAgent):

    def __init__(self,
            agent_name='inspiration_retriever_reflection',
            table_name='inspirations_040611',
            db_dir='database/ChromDB/inspirations',
            inspirations_path='database/inspirations/inspirations_040611.json',
            log_dir=None,
            llm_log_dir=None
        ) -> None:
        super().__init__(agent_name,llm_log_dir)
        self.retriever=InspirationRetriever(
            table_name=table_name,
            db_dir=db_dir
        )
        if not os.path.isdir(db_dir) or len(os.listdir(db_dir))==1:
            self.retriever.create_table(inspirations_path)
        self.anno_reflection_path = os.path.join(log_dir,'anno_research_reflection.jsonl')
        self.anno_mog_path = os.path.join(log_dir,'mog.json')
        with open(inspirations_path,'r') as f:
            ds = json.load(f)
        path = os.path.join(llm_log_dir,'inspiration_label.json')
        if not os.path.isfile(path):
            for d in ds:
                d['used'] = False
            with open(path,'w') as f:
                json.dump(ds,f,indent='\t')
        self.insp_path = path

    def load_annos(self):
        with open(self.insp_path,'r') as f:
            ds = json.load(f)
        annos = []
        for d in ds:
            if not d['used']:
                annos.append(d)
        return annos

    def append_anno(self,anno,path):
        with open(path,'a') as f:
            f.write(json.dumps(anno)+'\n')

    def load_block_names(self):
        ns = []
        if os.path.isfile(self.anno_reflection_path):
            with open(self.anno_reflection_path,'r') as f:
                ds=f.readlines()
                ds=[json.loads(d) for d in ds]
                for d in ds:
                    ns.append(d['block_name'])
        return ns
    
    def load_reflections(self,key='inspiration'):
        poss,negs=[],[]
        if not os.path.isfile(self.anno_reflection_path):
            return poss, negs
        with open(self.anno_reflection_path,'r') as f:
            ds=f.readlines()
            ds=[json.loads(d) for d in ds]
            for d in ds:
                if d['type']=='pos':
                    poss.append(d[key])
                elif d['type']=='neg':
                    negs.append(d[key])
                else:
                    raise NotImplementedError
        return poss,negs

    def set_used(self,insp_ids):
        self.retriever.set_used(insp_ids)


    def __call__(self, num=10, **kwargs):
        self.reflect()
        insps_random = self.load_annos()
        poss,negs = self.load_reflections()
        insps = {}
        for pos in poss:
            insp = self.retriever(query=pos,num=5)
            insps.update(insp)
        for neg in negs:
            insp = self.retriever(query=neg,num=5,reverse=True)
            insps.update(insp)
        if len(insps) < num:
            insps_random = []
            if hasattr(self, "load_annos") and callable(self.load_annos):
                try:
                    insps_random = self.load_annos()  # [{id, inspiration, used...}, ...]
                except Exception:
                    insps_random = []

            if insps_random:
                take = min(len(insps_random), num * 3)
                picks = np.random.choice(insps_random, size=take, replace=False)
                for d in picks:
                    insps[str(d['id'])] = d['inspiration']

        # 4) 최종 반환 (가능한 만큼)
        if len(insps) == 0:
            return {}
        ids = list(insps.keys())
        take = min(len(ids), num)
        pick = np.random.choice(ids, size=take, replace=False)
        return {i: insps[i] for i in pick}

    def reflect(self):
        bns = self.load_block_names()
        assert os.path.isfile(self.anno_mog_path),self.anno_mog_path
        with open(self.anno_mog_path,'r') as f:
            ds = json.load(f)
        poss,negs = [],[]
        for k,v in ds.items():
            if k in bns or v['iter']==0:
                continue
            anno = {
                'iter':v['iter'],
                'type':None,
                'block_name':k,
                'block':v['blocks'][0],
                'from_block_name':v['from_block_name'],
                'from_block':ds[v['from_block_name']]['blocks'][0],
                'inspiration_id':v['inspiration_id'],
                'inspiration':self.retriever[v['inspiration_id']],
                'acc':v['acc'],
                'from_acc':ds[v['from_block_name']]['acc']
            }
            if anno['acc']>anno['from_acc']:
                anno['type'] = 'pos'
                poss.append(anno)
            else:
                anno['type'] = 'neg'
                negs.append(anno)
            self.append_anno(anno,self.anno_reflection_path)
        # for pos in poss+negs:
        #     input = f"""<block>{pos['from_block']}</block>\n<block>{pos['block']}</block>"""
        #     prompt = PROMPT_RESEARCH_REFLECTION_INSPIRATION.format(input=input)
        #     res = super().__call__(prompt)
        #     pdb.set_trace()
        #     out = res['output']
        #     if out:
        #         pos['reflection'] = out
        #         self.append_anno(pos,self.anno_reflection_path)
        
        res = {'pos_num':len(poss),'neg_num':len(negs)}
        return res

    def parse_reflection(self,txt):
        s = re.findall("<response>(.*)</response>",txt,re.DOTALL)
        if len(s)>0:
            return s[0].strip()
        return None




class InspirationSampler():


    def __init__(self, insp_path='',log_dir='') -> None:
        with open(insp_path,'r') as f:
            ds = json.load(f)
        path = os.path.join(log_dir,'inspiration_label.json')
        if not os.path.isfile(path):
            for d in ds:
                d['used'] = False
            with open(path,'w') as f:
                json.dump(ds,f,indent='\t')
        self.insp_path = path

    def load_annos(self):
        with open(self.insp_path,'r') as f:
            ds = json.load(f)
        annos = []
        for d in ds:
            if not d['used']:
                annos.append(d)
        return annos
    
    def set_used(self,insp_ids):
        if isinstance(insp_ids,int):
            insp_ids = [str(insp_ids)]
        elif isinstance(insp_ids,str):
            insp_ids = [insp_ids]
        else:
            insp_ids = [str(id) for id in insp_ids]
        with open(self.insp_path,'r') as f:
            ds = json.load(f)
        for d in ds:
            if str(d['id']) in insp_ids:
                d['used'] = True
        with open(self.insp_path,'w') as f:
            json.dump(ds,f,indent='\t')


    def __call__(self, num=10, **kwargs):
        insps = self.load_annos()
        assert len(insps)>num,len(insps)
        insps = np.random.choice(insps,num,replace=False)
        res = {}
        for insp in insps[:num//2]:
            res[str(insp['id'])] = insp['inspiration']
        return res