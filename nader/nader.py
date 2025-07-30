import json
import os
import warnings
import logging
import shutil


from team_develop import TeamDevelop
from team_research import TeamResearch,TeamResearchNoReader,TeamResearchHandCraft
from tools.utils import *
from train_utils.train_templates import *

httpx_logger = logging.getLogger("httpx")
httpx_logger.propagate = False
httpx_logger = logging.getLogger("chromadb.telemetry.product.posthog")
httpx_logger.propagate = False
warnings.filterwarnings("ignore")


class Nader:

    def __init__(self,
                database_dir='data',
                dataset='cifar10',
                base_block=None,
                inspiration_retriever_mode='random',
                candiate_inspiration_num=10,
                inspirations_path='data/inspirations/inspirations_040611.json',
                research_team_name='nader',
                develop_team_name='',
                research_use_experience='research_fail_240807_experience',
                research_experience_mode='VDB',
                develop_use_experiecne='develop_allfailed_240709_experience',
                develop_experience_mode='VDB',
                log_dir = 'logs/trail1',
                train_log_dir='output/imagenet',
                research_mode=None,
                proposer_mode='llm',
                width=5,
                max_try=5,
                tag_prefix_base='resnet_trail1',
                mode='nas-bench',
                layers_num=20) -> None:
        self.base_block = base_block
        self.database_dir = database_dir
        self.dataset = dataset
        self.width = width
        self.max_try = max_try
        self.tag_prefix_base = tag_prefix_base
        self.log_dir = log_dir
        self.train_log_dir = train_log_dir
        self.mode = mode
        self.layers_num = layers_num
        code_dir = os.path.join(log_dir,'codes')
        self.code_dir = code_dir

        # blocks
        flag = False
        self.block_txt_dir = os.path.join(log_dir,'block_txt')
        os.makedirs(self.block_txt_dir,exist_ok=True)
        self.blocks_path = os.path.join(log_dir,'blocks.jsonl')
        if not os.path.isfile(self.blocks_path):
            path = os.path.join(self.block_txt_dir,f'{base_block}.txt')
            if not os.path.isfile(path):
                flag = True
                shutil.copy(os.path.join(self.database_dir,'blocks','txts',f'{base_block}.txt'),path)
            self.append_anno({'iter':0,'block_name':base_block,'raw_block_name':None,'inspiration_id':None},self.blocks_path)

        # logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(tag_prefix_base)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(log_dir,'log.txt'))
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        # logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.propagate = False
        self.logger = logger

        # develop team
        self.develop = TeamDevelop(
            team_name=develop_team_name,
            dataset=dataset,
            use_experience=develop_use_experiecne,
            experience_mode=develop_experience_mode,
            database_dir=database_dir,
            code_dir=code_dir,
            log_dir=log_dir,
            max_try=max_try,
            tag_prefix=tag_prefix_base,
            block_txt_dir=self.block_txt_dir,
            cell_mode=mode,
            layers_num=layers_num
        )

        # research team
        if research_team_name=='nader':
            self.research = TeamResearch(
                base_block=base_block,
                retriever_mode=inspiration_retriever_mode,
                candiate_inspiration_num=candiate_inspiration_num,
                inspirations_path=inspirations_path,
                mode=research_mode,
                proposer_mode=proposer_mode,
                use_experience=research_use_experience,
                experience_mode=research_experience_mode,
                log_dir=log_dir,
                tag_prefix=tag_prefix_base,
                block_txt_dir=self.block_txt_dir,
                block_anno_path=self.blocks_path,
                train_log_dir=train_log_dir
            )
        elif research_team_name=='nader_wor':
            self.research = TeamResearchNoReader(
                base_block=base_block,
                mode=research_mode,
                use_experience=research_use_experience,
                log_dir=log_dir,
                tag_prefix=tag_prefix_base,
                block_txt_dir=self.block_txt_dir,
                block_anno_path=self.blocks_path,
                train_log_dir=train_log_dir
            )
        elif research_team_name=='nader_hc':
            self.research = TeamResearchHandCraft(
                base_block=base_block,
                mode=research_mode,
                use_experience=research_use_experience,
                log_dir=log_dir,
                tag_prefix=tag_prefix_base,
                block_txt_dir=self.block_txt_dir,
                block_anno_path=self.blocks_path,
                train_log_dir=train_log_dir
            )
        else:
            raise NotImplementedError


        if flag:
            self.develop.block_gen.add_blocks_from_txt_dir(self.block_txt_dir)
            self.develop.model_gen.generate_all()

    
    def append_anno(self,anno,path):
        with open(path,'a') as f:
            f.write(json.dumps(anno)+'\n') 

    
    def run_one_iter(self,iter,width=None):
        if not width:
            width = self.width
        num = 0
        costs = {'research':{'prompt_tokens':0,'completion_tokens':0,'price':0},'develop':{'prompt_tokens':0,'completion_tokens':0,'price':0},'total':{'prompt_tokens':0,'completion_tokens':0,'price':0}}
        while num<width:
            research_res = self.research(num=self.width,iter=iter)
            for key in ['prompt_tokens','completion_tokens']:
                costs['research'][key] = research_res[key]
            proposals = research_res['proposals']
            for proposal in proposals:
                self.logger.info(f"User proposal:{proposal['block_name']}-{proposal['inspiration_id']}")
                model = self.develop(**proposal)
                for key in ['prompt_tokens','completion_tokens']:
                    costs['develop'][key] = model[key]
                if proposal['inspiration_id']!=-1:
                    self.research.set_used(proposal['inspiration_id'])
                if model['status'] and not model['existed']:
                    anno = {
                        'iter':iter,
                        'block_name':model['model_name'],
                        'raw_block_name':proposal['block_name'],
                        'inspiration_id':proposal['inspiration_id']
                    }
                    self.append_anno(anno,self.blocks_path)
                    num+=1
                    if num>=width:
                        break
        for key in ['prompt_tokens','completion_tokens']:
            costs['total'][key] = costs['research'][key] + costs['develop'][key]
        for key in ['research','develop','total']:
            costs[key]['price'] = costs[key]['prompt_tokens']/1e6*2.5 + costs[key]['completion_tokens']/1e6*10
        costs = {
            'iter':iter,
            'width':width,
            'costs':costs
        }
        with open(os.path.join(self.log_dir,'costs_gpt.jsonl'),'a') as f:
            f.write(json.dumps(costs)+'\n') 
        return costs

    def generate_train_script(self,task_root_dir=None,cluster='l40',batch_path=None):
        if not task_root_dir:
            task_root_dir = os.path.join(self.log_dir,'tasks')
        os.makedirs(task_root_dir,exist_ok=True)
        if not batch_path:
            batch_path = 'train_batch.sh'
        
        if self.mode=='nas-bench':
            template = NB201_TRAIN_TEMPLATE_MAP[cluster.lower()][self.dataset]
        elif cluster.lower()=='4090d' and self.dataset=='cifar10' and self.mode=='darts' and self.layers_num==8:
            template = TRAIN_DARTS_CIFAR10_LAYER8_4090D
        else:
            raise NotImplementedError
        template_all = TRAIN_BATCH
        models = []
        task_dir = os.path.join(task_root_dir,self.dataset,self.tag_prefix_base)
        os.makedirs(task_dir,exist_ok=True)
        files = os.listdir(self.train_log_dir)
        new_dirs = []
        with open(self.blocks_path,'r') as f:
            annos = f.readlines()
            annos = [json.loads(anno) for anno in annos]
        for anno in annos:
            if anno['block_name'] not in files:
                models.append(anno['block_name'])

        if len(models)>0:
            for model in models:
                path = os.path.join(task_dir,f"{model}.sh")
                txt = template.format(job_name=model[-10:],model_name=model,code_dir=self.code_dir,train_log_dir=self.train_log_dir)
                with open(path,'w') as f:
                    f.write(txt)
            models_txt = [i+'.sh' for i in models]
            models_txt = ' '.join(models_txt)
            models_txt = f'({models_txt})'
            txt = template_all.format(task_dir=task_dir,models=models_txt)
            with open(batch_path,'w') as f:
                f.write(txt)
        return models
