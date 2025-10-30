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
                candidate_inspiration_num=10,
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
                width=5,   # 한 iteration에서 수용할 새 모델 개수 목표
                max_try=5, #modifier 재시도 횟수 제한 
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
        
        # Always ensure base_block file exists in block_txt_dir
        base_block_path = os.path.join(self.block_txt_dir, f'{base_block}.txt')
        base_block_src = os.path.join(self.database_dir, 'blocks', 'txts', f'{base_block}.txt')
        
        if not os.path.isfile(base_block_path):
            if os.path.isfile(base_block_src):
                print(f"Copying base block: {base_block_src} -> {base_block_path}")
                shutil.copy(base_block_src, base_block_path)
                flag = True
            else:
                raise FileNotFoundError(f"Base block file not found: {base_block_src}")
        
        # Initialize blocks.jsonl if it doesn't exist
        if not os.path.isfile(self.blocks_path):
            self.append_anno({'iter':0,'block_name':base_block,'raw_block_name':None,'inspiration_id':None},self.blocks_path)
            flag = True  # blocks.jsonl이 새로 생성되면 블록도 등록해야 함
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
                base_block=base_block, # retinanet
                retriever_mode=inspiration_retriever_mode, # random
                candidate_inspiration_num=candidate_inspiration_num, # 10
                inspirations_path=inspirations_path, # data/inspirations/inspirations_040611.json
                mode=research_mode, # dfs-one
                proposer_mode=proposer_mode, # llm
                use_experience=research_use_experience, # research_fail_240807_experience
                experience_mode=research_experience_mode, # VDB
                log_dir=log_dir, # logs/detection-bench/coco2017/nader_llm_retinanet_random_dfs-width5_run10_seed888
                tag_prefix=tag_prefix_base, # nader_llm_retinanet_seed888
                block_txt_dir=self.block_txt_dir, # logs/detection-bench/coco2017/nader_llm_retinanet_random_dfs-width5_run10_seed888/block_txt
                block_anno_path=self.blocks_path, # logs/detection-bench/coco2017/nader_llm_retinanet_random_dfs-width5_run10_seed888/blocks.jsonl
                train_log_dir=train_log_dir # output/detection-bench/coco2017
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
        
        # Initialize blocks based on mode
        if self.mode == 'detection':
            # For detection mode, always register blocks from sections to preserve section names
            blocks_base_dir = os.path.join(code_dir, 'blocks', 'base')
            block_mapping_path = os.path.join(blocks_base_dir, 'block_mapping.json')
            
            # Register blocks if mapping doesn't exist or if this is initial setup
            if not os.path.isfile(block_mapping_path) or flag:
                print(f"Registering detection blocks from: {base_block_path}")
                from ModelFactory.block_factory import BlockFactory
                bf = BlockFactory(
                    blocks_dir=blocks_base_dir,
                    type='base',
                    register_path='ModelFactory.register',
                    mode='detection'
                )
                result = bf.add_blocks_from_sections_path(base_block_path, id_prefix=base_block)
                if isinstance(result, dict) and 'error' in result:
                    raise RuntimeError(f"Failed to register blocks: {result['error']}")
                print(f"Successfully registered {len(result)} FPN sections")
            else:
                print(f"Detection blocks already registered (found {block_mapping_path})")
        elif flag:
            # For NAS-Bench mode, only initialize if flag is True
            self.develop.block_gen.add_blocks_from_txt_dir(self.block_txt_dir)
            self.develop.model_gen.generate_all()

    def append_anno(self,anno,path):
        with open(path,'a') as f:
            f.write(json.dumps(anno)+'\n') 
# iter 는 매 실행마다 1 2 3, width는 5 
    def run_one_iter(self,iter,width=None):
        if not width: 
            width = self.width # 5 
        num = 0
        costs = {'research':{'prompt_tokens':0,'completion_tokens':0,'price':0},'develop':{'prompt_tokens':0,'completion_tokens':0,'price':0},'total':{'prompt_tokens':0,'completion_tokens':0,'price':0}}
        while num<width: # 5개 채울때 까지 반복 
            research_res = self.research(num=self.width,iter=iter)
            for key in ['prompt_tokens','completion_tokens']:
                costs['research'][key] = research_res[key]
            proposals = research_res['proposals'] # 연구팀이 제안 생성 
            for proposal in proposals:
                self.logger.info(f"User proposal:{proposal['block_name']}-{proposal['inspiration_id']}")
                print(f"개발팀이 제안을 받아서 모델을 생성 중")
                model = self.develop(**proposal) # 개발팀이 제안을 받아 모델 생성 
                print(f"개발팀이 제안을 받아서 모델 하나 생성 완료")
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
    #새로 생성된 모델들을 학습시키기 위한 .sh 스크립트(개별 + 배치)를 자동으로 만들어주는 함수
    def generate_train_script(self,task_root_dir=None,cluster='l40',batch_path=None): #batch_path: train_full_iter1.sh
        if not task_root_dir: #상위에서 없어서 로그 하위에 tasks 폴더 생성 
            task_root_dir = os.path.join(self.log_dir,'tasks')
        os.makedirs(task_root_dir,exist_ok=True)
        if not batch_path:
            batch_path = 'train_batch.sh'
        
        if self.mode=='nas-bench':
            template = NB201_TRAIN_TEMPLATE_MAP[cluster.lower()][self.dataset]
        elif cluster.lower()=='4090d' and self.dataset=='cifar10' and self.mode=='darts' and self.layers_num==8:
            template = TRAIN_DARTS_CIFAR10_LAYER8_4090D
        elif self.mode=='detection':
            template = DETECTION_TRAIN_TEMPLATE_MAP[cluster.lower()][self.dataset]
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
