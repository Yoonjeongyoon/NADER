import os
import warnings
import subprocess
import time
import argparse

from tools.utils import *
from nader import Nader


warnings.filterwarnings("ignore")

#결과 로그 출력 함수 
def print_and_save(text, file):
    print(text) #입력받은 텍스트를 화면에 출력 
    with open(file, 'a') as f: #파일을 append모드로 열고 뒤에 덧붙임 
        f.write(text + '\n') #한줄 적고 줄바꿈 
        f.flush()#파일 버퍼 삭제 

blockmap={
    "resnet_basic":{777:"resnet_basic",888:"resnet_basic",999:"resnet_basic"}, #resnet_basic 블록 3개 
    "nasbench_random":{
        777:"nasbench201_seed777", #nasbench201_seed777 블록 
        888:"resnet_basic", #resnet_basic 블록 
        999:"nasbench201_seed777" #nasbench201_seed777 블록 
    }
}

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--max-iter',type=int,required=True)
    parser.add_argument('-d','--dataset',type=str,choices=['cifar10','cifar100','imagenet16-120'],required=True)
    parser.add_argument('-b','--base-block',type=str,default='resnet_basic',choices=['resnet_basic','convnext','nasbench_random'])
    parser.add_argument('-r','--research-team-name',type=str,default='nader',choices=['nader','nader_wor','nader_hc'])
    parser.add_argument('-c','--cluster',type=str,default='local',choices=['l40','a800','4090d','local'])
    parser.add_argument('--seed',type=int,default=777,choices=[777,888,999])
    parser.add_argument('--width',default=5,type=int)
    parser.add_argument('--inspiration-retriever-mode',default='random',type=str,choices=['random','reflection']) #Reflector on off 
    parser.add_argument('--research-experience',default=None,type=str,choices=[None,'research_fail_240807_experience'])
    parser.add_argument('-p','--proposer-mode',type=str,default='llm')
    parser.add_argument('--log-dir',type=str,default='logs/nas-bench-201')
    parser.add_argument('--train-log-dir',type=str,default='output/nas-bench-201')

    args = parser.parse_args()
    cluster = args.cluster
    seed = args.seed
    max_iter = args.max_iter
    width = args.width
    dataset = args.dataset # 기본 cifar10
    base_block=args.base_block # 기본 resnet_basic
    research_team_name=args.research_team_name # abulation 용 
    research_experience = args.research_experience
    inspiration_retriever_mode=args.inspiration_retriever_mode

    tag_prefix_base=f'{research_team_name}_{args.proposer_mode}_{base_block}_seed{seed}' # nader_llm_resnet_basic_seed888
    log_dir = f'{args.log_dir}/{dataset}/{research_team_name}_{args.proposer_mode}_{base_block}_{inspiration_retriever_mode}_dfs-width{args.width}_run{args.max_iter}_seed{seed}'
    set_seed(seed)
    nader = Nader( #Nader 객체 초기화 
        base_block=blockmap[base_block][seed],
        dataset=f"nas-bench-201-{dataset}",
        database_dir='data/nas-bench-201',
        inspiration_retriever_mode=inspiration_retriever_mode, # proposer가 경험데이터 베이스 E를 사용할지 안할지 
        research_team_name=research_team_name,
        develop_team_name='try-fb',
        research_use_experience=research_experience,
        develop_use_experiecne='develop_allfailed_240709_experience',
        log_dir=log_dir,
        train_log_dir=f'{args.train_log_dir}/{dataset}', # output/nas-bench-201/cifar10
        tag_prefix_base=tag_prefix_base,# nader_llm_resnet_basic_seed888
        research_mode='dfs-one',
        proposer_mode=args.proposer_mode
    )

    best_test_acc,best_val_acc=-1,-1
    runs_log_path = os.path.join(log_dir,'log_runs.log') #로그 파일 경로 logs/nas-bench-201/cifar10/nader_llm_resnet_basic_random_dfs-width5_run10_seed888/log_runs.log
    for iter in range(1,max_iter+1): # 1번 반복
        start = time.time() # 시간 측정
        print_and_save(f"Iter: {iter}/{max_iter}",runs_log_path) # 반복 횟수 출력
        
        # create model
        nader.run_one_iter(iter=iter,width=width) #research 에서 inspiration 받아서 그래프 생성 + 모델 생성 -> 최대 5개 
        path = os.path.join(log_dir,f"train_full_iter{iter}.sh")
        models = nader.generate_train_script(cluster=cluster,batch_path=path) # 모델들 리스트랑 실행 파일 생성  생성된 모델들을 학습시키기 위한 bash 스크립트를 만
        print_and_save(f"Create models:{models}",runs_log_path)
        log_dirs = [os.path.join(nader.train_log_dir,model,'1') for model in models] # output/nas-bench-201/cifar10/resnet_basic/1
        print_and_save(f"train_log_dir: {nader.train_log_dir}",runs_log_path)
        print_and_save(f"models: {models}",runs_log_path)
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
                ns.append(f"{n}/200")
                path = os.path.join(d,'train_status.txt')
                if not os.path.isfile(path):
                    status=False
            print_and_save(f"Process:{ns}",runs_log_path)
            if status:
                break
            time.sleep(60)
        print_and_save(f"Training finised!",runs_log_path)
        e2 = time.time()

        # summary
        block_list,test_acc = nader.research.mog_graph_manage.update_train_result(nader.train_log_dir,filename='test_acc',tag_prefix=tag_prefix_base,iter=iter,flush=True)
        block_list,val_acc = nader.research.mog_graph_manage.update_train_result(nader.train_log_dir,filename='val_acc',tag_prefix=tag_prefix_base,anno_name='mog_val',iter=iter,flush=True)
        best_test_acc = max(test_acc,best_test_acc)
        best_val_acc = max(val_acc,best_val_acc)
        print_and_save(f"Iter {iter}/{max_iter}, Val acc:{val_acc}, Test acc:{test_acc}, Best val acc{best_val_acc}, Best test acc:{best_test_acc}",runs_log_path)