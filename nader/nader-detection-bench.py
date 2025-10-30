import os
import warnings
import subprocess
import time
import argparse
from datetime import datetime

from tools.utils import *
from nader import Nader


warnings.filterwarnings("ignore")

#결과 로그 출력 함수 
def print_and_save(text, file):
    print(text) #입력받은 텍스트를 화면에 출력 
    with open(file, 'a') as f: #파일을 append모드로 열고 뒤에 덧붙임 
        f.write(text + '\n') #한줄 적고 줄바꿈 
        f.flush()#파일 버퍼 삭제 

def get_blockmap(fpn_version='simple'):
    """Get block mapping based on FPN version choice.
    
    Args:
        fpn_version: 'simple' (default, single-input blocks) or 'full' (multi-input blocks, experimental)
    """
    fpn_block_name = f"neck_fpn_{fpn_version}" if fpn_version == 'simple' else "neck_fpn"
    
    return {
        "resnet_basic":{777:"resnet_basic",888:"resnet_basic",999:"resnet_basic"},
        "nasbench_random":{
            777:"nasbench201_seed777",
            888:"resnet_basic",
            999:"nasbench201_seed777"
        },
        "neck_fpn":{
            777: fpn_block_name,
            888: fpn_block_name,
            999: fpn_block_name
        },
        "fpn_nasbench":{
            777:"fpn_nasbench",
            888:"fpn_nasbench",
            999:"fpn_nasbench"
        }
    }

if __name__=='__main__':
#baseblock 지정해야함 코드 살펴봐야함 
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--max-iter',type=int,required=True)
    parser.add_argument('-d','--dataset',type=str,choices=['cifar10','cifar100','imagenet16-120','coco2017'],required=True)
    parser.add_argument('-b','--base-block',type=str,default='fpn_nasbench',choices=['resnet_basic','convnext','nasbench_random','retinanet','neck_fpn','fpn_nasbench'])
    parser.add_argument('-r','--research-team-name',type=str,default='nader',choices=['nader','nader_wor','nader_hc'])
    parser.add_argument('-c','--cluster',type=str,default='local',choices=['l40','a800','4090d','local'])
    parser.add_argument('--seed',type=int,default=777,choices=[777,888,999])
    parser.add_argument('--width',default=5,type=int)
    parser.add_argument('--inspiration-retriever-mode',default='random',type=str,choices=['random','reflection']) #Reflector on off 
    parser.add_argument('--research-experience',default=None,type=str,choices=[None,'research_fail_240807_experience'])
    parser.add_argument('-p','--proposer-mode',type=str,default='llm')
    parser.add_argument('--log-dir',type=str,default='logs/detection-bench')
    parser.add_argument('--train-log-dir',type=str,default='output/detection-bench')
    parser.add_argument('--fpn-version',type=str,default='simple',choices=['simple','full'],
                        help='FPN block version: "simple" (single-input blocks, recommended) or "full" (multi-input blocks, experimental)')

    args = parser.parse_args()
    cluster = args.cluster
    seed = args.seed
    max_iter = args.max_iter
    width = args.width
    dataset = args.dataset # coco2017
    base_block=args.base_block # 기본 resnet_basic
    research_team_name=args.research_team_name # abulation 용 
    research_experience = args.research_experience
    inspiration_retriever_mode=args.inspiration_retriever_mode
    fpn_version = args.fpn_version  # 'simple' or 'full'

    # Get blockmap based on FPN version
    blockmap = get_blockmap(fpn_version)
    
    tag_prefix_base=f'{research_team_name}_{args.proposer_mode}_{base_block}_seed{seed}' # nader_llm_retinanet_seed888
    log_dir = f'{args.log_dir}/{dataset}/{research_team_name}_{args.proposer_mode}_{base_block}_{inspiration_retriever_mode}_dfs-width{args.width}_run{args.max_iter}_seed{seed}'
    set_seed(seed)
    
    print(f"="*80)
    print(f"NADER Detection Benchmark Configuration")
    print(f"="*80)
    print(f"Dataset: {dataset}")
    print(f"Base Block: {base_block}")
    print(f"FPN Version: {fpn_version} ({blockmap[base_block][seed]})")
    print(f"Seed: {seed}")
    print(f"Max Iterations: {max_iter}")
    print(f"Width: {width}")
    print(f"="*80)
    
    nader = Nader( #Nader 객체 초기화 
        base_block=blockmap[base_block][seed],
        dataset=f"detection-bench-{dataset}",
        database_dir='data/detection-bench',
        inspiration_retriever_mode=inspiration_retriever_mode, # proposer가 경험데이터 베이스 E를 사용할지 안할지 
        research_team_name=research_team_name,
        develop_team_name='try-fb',
        research_use_experience=research_experience,
        develop_use_experiecne='develop_allfailed_240709_experience',
        log_dir=log_dir,
        train_log_dir=f'{args.train_log_dir}/{dataset}', # output/detection-bench/coco2017
        tag_prefix_base=tag_prefix_base,# nader_llm_retinanet_seed888
        research_mode='dfs-one',
        proposer_mode=args.proposer_mode,   
        mode='detection'
    )

    best_test_acc,best_val_acc=-1,-1
    runs_log_path = os.path.join(log_dir,'log_runs.log') #로그 파일 경로 logs/detection-bench/coco2017/nader_llm_retinanet_random_dfs-width5_run10_seed888/log_runs.log
    for iter in range(1,max_iter+1): # 1번 반복
        start = time.time() # 시간 측정
        print_and_save(f"Iter: {iter}/{max_iter}",runs_log_path) # 반복 횟수 출력
        
        # create model
        nader.run_one_iter(iter=iter,width=width) #research 에서 inspiration 받아서 그래프 생성 + 모델 생성 -> 최대 5개 
        path = os.path.join(log_dir,f"train_full_iter{iter}.sh")
        models = nader.generate_train_script(cluster=cluster,batch_path=path) # 모델들 리스트랑 실행 파일 생성  생성된 모델들을 학습시키기 위한 bash 스크립트를 만
        print_and_save(f"Create models:{models}",runs_log_path)
        # Use realpath to match template behavior
        log_dirs = [os.path.realpath(os.path.join(nader.train_log_dir,model,'1')) for model in models]
        print_and_save(f"train_log_dir: {nader.train_log_dir}",runs_log_path)
        print_and_save(f"models: {models}",runs_log_path)
        print_and_save(f"log_dirs (realpath): {log_dirs}",runs_log_path)
        e1 = time.time() # 시간 측정
        
        process = subprocess.Popen(
            f"bash {path}",
            shell=True,
            stdout=subprocess.DEVNULL,  
            stderr=subprocess.STDOUT    
        )
        while len(log_dirs)>0:
            ns = []
            status = True
            for d in log_dirs:
                # For detection mode, check MMDetection training progress
                if dataset == 'coco2017':
                    # Check MMDetection work_dir for training progress
                    work_dir = os.path.join(d, 'work_dir')
                    
                    # Find log file (MMDetection creates timestamped subdirectories)
                    log_file = None
                    n = 0
                    
                    # Try to find timestamped log directories
                    if os.path.exists(work_dir):
                        subdirs = [os.path.join(work_dir, d) for d in os.listdir(work_dir) 
                                   if os.path.isdir(os.path.join(work_dir, d))]
                        subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                        
                        for subdir in subdirs:
                            # Try .log.json first, then .log
                            potential_logs = [
                                os.path.join(subdir, f'{os.path.basename(subdir)}.log.json'),
                                os.path.join(subdir, f'{os.path.basename(subdir)}.log')
                            ]
                            for potential_log in potential_logs:
                                if os.path.isfile(potential_log):
                                    log_file = potential_log
                                    break
                            if log_file:
                                break
                    
                    if log_file:
                        print_and_save(f"Found MMDetection log: {log_file}", runs_log_path)
                        # Parse log to get current epoch
                        try:
                            if log_file.endswith('.log.json'):
                                with open(log_file, 'r') as f:
                                    lines = f.readlines()
                                    if lines:
                                        last_line = json.loads(lines[-1])
                                        epoch = last_line.get('epoch', 0)
                                        n = epoch
                                        print_and_save(f"Current epoch: {epoch}/12", runs_log_path)
                            else:  # .log file
                                # Parse text log for epoch info
                                with open(log_file, 'r') as f:
                                    content = f.read()
                                    import re
                                    # MMDetection format: "Epoch(train)  [1][50/1250]"
                                    epochs = re.findall(r'Epoch\([^)]+\)\s*\[(\d+)\]', content)
                                    if epochs:
                                        n = int(epochs[-1])
                                        print_and_save(f"Current epoch: {n}/12 (from .log)", runs_log_path)
                        except Exception as e:
                            print_and_save(f"Error parsing log: {e}", runs_log_path)
                            n = 0
                    else:
                        print_and_save(f"MMDetection log not found in: {work_dir}", runs_log_path)
                        n = 0
                    
                    # Check if training is complete by looking for train_status.txt
                    status_file = os.path.join(d, 'train_status.txt')
                    print_and_save(f"Checking status file: {status_file}", runs_log_path)
                    print_and_save(f"Status file exists: {os.path.isfile(status_file)}", runs_log_path)
                    
                    if os.path.isfile(status_file):
                        with open(status_file, 'r') as f:
                            status_content = f.read().strip()
                        print_and_save(f"Status file content: '{status_content}'", runs_log_path)
                        
                        if status_content == 'done':
                            print_and_save(f" Training completed: {status_file}", runs_log_path)
                            n = 12  # Training completed
                        elif status_content == 'failed':
                            print_and_save(f" Training failed: {status_file}", runs_log_path)
                            n = 12  # Mark as completed but failed
                        elif status_content == 'interrupted':
                            print_and_save(f" Training interrupted: {status_file}", runs_log_path)
                            n = 12  # Mark as completed but interrupted
                        elif status_content == 'unknown':
                            print_and_save(f" Training status unknown: {status_file}", runs_log_path)
                            n = 12  # Mark as completed but unknown
                        else:
                            print_and_save(f"Training in unexpected state: {status_content}", runs_log_path)
                            n = 0
                    else:
                        # Debug: Check if directory exists and list its contents
                        if os.path.exists(d):
                            dir_contents = os.listdir(d)
                            print_and_save(f"Directory exists, contents: {dir_contents}", runs_log_path)
                        else:
                            print_and_save(f"Directory does not exist: {d}", runs_log_path)
                        # Fallback: Check if training is still in progress by looking at log
                        if log_file and os.path.isfile(log_file):
                            try:
                                if log_file.endswith('.log.json'):
                                    with open(log_file, 'r') as f:
                                        lines = f.readlines()
                                        if lines:
                                            last_line = json.loads(lines[-1])
                                            epoch = last_line.get('epoch', 0)
                                            n = epoch
                                            print_and_save(f"Training in progress, current epoch: {epoch}/12", runs_log_path)
                                else:
                                    with open(log_file, 'r') as f:
                                        content = f.read()
                                        import re
                                        # MMDetection format: "Epoch(train)  [1][50/1250]"
                                        epochs = re.findall(r'Epoch\([^)]+\)\s*\[(\d+)\]', content)
                                        if epochs:
                                            n = int(epochs[-1])
                                            print_and_save(f"Training in progress, current epoch: {n}/12", runs_log_path)
                            except Exception as e:
                                print_and_save(f"Error reading log: {e}", runs_log_path)
                                n = 0
                        else:
                            n = 0
                else:
                    # Original NAS-Bench logic for other datasets
                    path = os.path.join(d,'val_acc.txt')
                    print_and_save(f"Checking: {path}",runs_log_path)
                    if not os.path.isfile(path):
                        print_and_save(f"File not found: {path}",runs_log_path)
                        n = 0
                    else:
                        print_and_save(f"File found: {path}",runs_log_path)
                        with open(path,'r') as f:
                            ds = f.readlines()
                            n = len(ds)-1
                            print_and_save(f"File content lines: {len(ds)}, n={n}",runs_log_path)
                
                ns.append(f"{n}/12" if dataset == 'coco2017' else f"{n}/200")
                
                # Check training status (final check)
                path = os.path.join(d,'train_status.txt')
                if not os.path.isfile(path):
                    print_and_save(f" Status file not found yet: {path}", runs_log_path)
                    status=False
                else:
                    print_and_save(f" Status file found: {path}", runs_log_path)
                    
            print_and_save(f"Process:{ns}, All status files ready:{status}",runs_log_path)
            if status:
                print_and_save(f"All models completed training!", runs_log_path)
                break
            time.sleep(60)
        print_and_save(f"Training finised!",runs_log_path)
        e2 = time.time()

        # summary
        if dataset == 'coco2017':
            # For detection mode, use mAP instead of accuracy
            block_list,test_acc = nader.research.mog_graph_manage.update_train_result(nader.train_log_dir,filename='val_acc',tag_prefix=tag_prefix_base,iter=iter,flush=True)
            block_list,val_acc = nader.research.mog_graph_manage.update_train_result(nader.train_log_dir,filename='val_acc',tag_prefix=tag_prefix_base,anno_name='mog_val',iter=iter,flush=True)
            best_test_acc = max(test_acc,best_test_acc)
            best_val_acc = max(val_acc,best_val_acc)
            print_and_save(f"Iter {iter}/{max_iter}, COCO bbox mAP:{val_acc}, Best mAP:{best_val_acc}",runs_log_path)
        else:
            # Original NAS-Bench logic
            block_list,test_acc = nader.research.mog_graph_manage.update_train_result(nader.train_log_dir,filename='test_acc',tag_prefix=tag_prefix_base,iter=iter,flush=True)
            block_list,val_acc = nader.research.mog_graph_manage.update_train_result(nader.train_log_dir,filename='val_acc',tag_prefix=tag_prefix_base,anno_name='mog_val',iter=iter,flush=True)
            best_test_acc = max(test_acc,best_test_acc)
            best_val_acc = max(val_acc,best_val_acc)
            print_and_save(f"Iter {iter}/{max_iter}, coco/bbox_mAP:{val_acc}, Test acc:{test_acc}, Best val acc{best_val_acc}, Best test acc:{best_test_acc}",runs_log_path)