TRAIN_BATCH = """directory="{task_dir}"
models={models}

cd ${{directory}}
for file in "${{models[@]}}"
do
    if [[ "$file" == *.sh ]]; then
        echo ${{file}}
        bash ${{file}}
        sleep 3
    fi
done
"""

TRAIN_IMAGENET_l40 = f"""srun -p l40s-mig \
            --workspace-id fdf02161-0d6e-4160-ae07-d95e0b94ce6d \
            -j {{job_name}} \
            -r N7lP.N.I80.8 \
            -N 1 \
            -d StandAlone \
            --framework pytorch \
            -a \
            -o logs/out_{{job_name}}.log \
            --container-image registry.cn-sh-01.sensecore.cn/devsft-ccr-0/yzk3:20240521-21h24m24s \
            --container-mounts 3cc34bef-b9bc-11ee-8090-9ea738d389d3:/mnt/afs \
            bash -c "ln -s /mnt/afs/yangzekang/envs/modelgen /usr/local/lib/miniconda3/envs/modelgen && source activate modelgen && NCCL_DEBUG=INFO && cd /mnt/afs/yangzekang/ModelGen/ModelGen && torchrun --nproc_per_node 8 train_imagenet.py --model_name {{model_name}} --tag 1 --warm-up-epochs 5 --epochs 50 --code-dir {{code_dir}}  --output {{train_log_dir}}"
        """

TRAIN_IMAGENET_a800 = f"""srun -p a64506b4-0df7-4eef-9f17-f283db665d87 \
            --workspace-id fdf02161-0d6e-4160-ae07-d95e0b94ce6d \
            -j {{job_name}} \
            -r N3sS.Ii.I60.8 \
            -N 1 \
            -d StandAlone \
            --framework pytorch \
            -a \
            -o logs/out_{{job_name}}.log \
            --container-image registry.cn-sh-01.sensecore.cn/devsft-ccr-0/yzk3:20240521-21h24m24s \
            --container-mounts de09d758-aac8-11ee-bc0d-9a07d96b4341:/mnt/afs \
            bash -c "source activate cloud-ai-lab && NCCL_DEBUG=INFO && cd /mnt/afs/yangzekang/ModelGen && torchrun --nproc_per_node 8 train_imagenet.py --model_name {{model_name}} --tag 1 --warm-up-epochs 5 --epochs 50 --code-dir {{code_dir}}  --output {{train_log_dir}}"
        """

TRAIN_NAS_BENCH_201_CIFAR10_l40 = f"""srun -p l40s-mig \
            --workspace-id fdf02161-0d6e-4160-ae07-d95e0b94ce6d \
            -j {{job_name}} \
            -r N7lP.N.I80.1 \
            -N 1 \
            -d StandAlone \
            --framework pytorch \
            -a \
            -o logs/out_{{job_name}}.log \
            --container-image registry.cn-sh-01.sensecore.cn/devsft-ccr-0/yzk3:20240521-21h24m24s \
            --container-mounts 3cc34bef-b9bc-11ee-8090-9ea738d389d3:/mnt/afs \
            bash -c "ln -s /mnt/afs/yangzekang/envs/modelgen /usr/local/lib/miniconda3/envs/modelgen && source activate modelgen && NCCL_DEBUG=INFO && cd /mnt/afs/yangzekang/ModelGen/ModelGen && python train_cifar10.py --model_name {{model_name}} --tag 1 --code-dir {{code_dir}} --output {{train_log_dir}}"
        """

TRAIN_NAS_BENCH_201_CIFAR100_l40 = f"""srun -p l40s-mig \
            --workspace-id fdf02161-0d6e-4160-ae07-d95e0b94ce6d \
            -j {{job_name}} \
            -r N7lP.N.I80.1 \
            -N 1 \
            -d StandAlone \
            --framework pytorch \
            -a \
            -o logs/out_{{job_name}}.log \
            --container-image registry.cn-sh-01.sensecore.cn/devsft-ccr-0/yzk3:20240521-21h24m24s \
            --container-mounts 3cc34bef-b9bc-11ee-8090-9ea738d389d3:/mnt/afs \
            bash -c "ln -s /mnt/afs/yangzekang/envs/modelgen /usr/local/lib/miniconda3/envs/modelgen && source activate modelgen && NCCL_DEBUG=INFO && cd /mnt/afs/yangzekang/ModelGen/ModelGen && python train_cifar100.py --model_name {{model_name}} --tag 1 --code-dir {{code_dir}} --output {{train_log_dir}}"
        """

TRAIN_NAS_BENCH_201_IMAGENET16_120_l40 = f"""srun -p l40s-mig \
            --workspace-id fdf02161-0d6e-4160-ae07-d95e0b94ce6d \
            -j {{job_name}} \
            -r N7lP.N.I80.1 \
            -N 1 \
            -d StandAlone \
            --framework pytorch \
            -a \
            -o logs/imagenet16-120/out_{{job_name}}.log \
            --container-image registry.cn-sh-01.sensecore.cn/devsft-ccr-0/yzk3:20240521-21h24m24s \
            --container-mounts 3cc34bef-b9bc-11ee-8090-9ea738d389d3:/mnt/afs \
            bash -c "ln -s /mnt/afs/yangzekang/envs/modelgen /usr/local/lib/miniconda3/envs/modelgen && source activate modelgen && NCCL_DEBUG=INFO && cd /mnt/afs/yangzekang/ModelGen/ModelGen && python train_imagenet16_120.py --model_name {{model_name}} --tag 1 --code-dir {{code_dir}} --output {{train_log_dir}}"
        """

TRAIN_NAS_BENCH_201_CIFAR10_4090D = f"""sco acp jobs create --workspace-name=tetras-share \
    --aec2-name=tetrasvideo4090d \
    --job-name=nb-c100_{{job_name}} \
    --container-image-url='registry.cn-sh-01.sensecore.cn/studio-aicl/ubuntu20.04-py3.10-cuda11.8-cudnn8-transformer4.30.0:v1.0.0-20240419-173534-a40fd64e1' \
    --storage-mount 53c47ef7-3457-11ef-ae95-ee12bf38b649:/mnt/afs \
    --training-framework=pytorch \
    --worker-nodes=1 \
    --worker-spec='N10lP.nn.A80a.1' \
    --command="ln -s /mnt/afs/yangzekang/envs/modelgen /usr/local/lib/miniconda3/envs/modelgen
source activate modelgen
cd /mnt/afs/yangzekang/ModelGen
python train_cifar10.py --model_name {{model_name}} --tag 1 --code-dir {{code_dir}} --output {{train_log_dir}}"
"""

TRAIN_NAS_BENCH_201_CIFAR100_4090D = f"""sco acp jobs create --workspace-name=tetras-share \
    --aec2-name=tetrasvideo4090d \
    --job-name=nb-c100_{{job_name}} \
    --container-image-url='registry.cn-sh-01.sensecore.cn/studio-aicl/ubuntu20.04-py3.10-cuda11.8-cudnn8-transformer4.30.0:v1.0.0-20240419-173534-a40fd64e1' \
    --storage-mount 53c47ef7-3457-11ef-ae95-ee12bf38b649:/mnt/afs \
    --training-framework=pytorch \
    --worker-nodes=1 \
    --worker-spec='N10lP.nn.A80a.1' \
    --command="ln -s /mnt/afs/yangzekang/envs/modelgen /usr/local/lib/miniconda3/envs/modelgen
source activate modelgen
cd /mnt/afs/yangzekang/ModelGen
python train_cifar100.py --model_name {{model_name}} --tag 1 --code-dir {{code_dir}} --output {{train_log_dir}}"
"""

TRAIN_NAS_BENCH_201_IMAGENET16_120_4090D = f"""sco acp jobs create --workspace-name=tetras-share \
    --aec2-name=tetrasvideo4090d \
    --job-name=nb-in_{{job_name}} \
    --container-image-url='registry.cn-sh-01.sensecore.cn/studio-aicl/ubuntu20.04-py3.10-cuda11.8-cudnn8-transformer4.30.0:v1.0.0-20240419-173534-a40fd64e1' \
    --storage-mount 53c47ef7-3457-11ef-ae95-ee12bf38b649:/mnt/afs \
    --training-framework=pytorch \
    --worker-nodes=1 \
    --worker-spec='N10lP.nn.A80a.1' \
    --command="ln -s /mnt/afs/yangzekang/envs/modelgen /usr/local/lib/miniconda3/envs/modelgen
source activate modelgen
cd /mnt/afs/yangzekang/ModelGen
python train_imagenet16_120.py --model_name {{model_name}} --tag 1 --code-dir {{code_dir}} --output {{train_log_dir}}"
"""


TRAIN_DARTS_CIFAR10_LAYER8_4090D = f"""sco acp jobs create --workspace-name=tetras-share \
    --aec2-name=tetrasvideo4090d \
    --job-name={{job_name}} \
    --container-image-url='registry.cn-sh-01.sensecore.cn/studio-aicl/ubuntu20.04-py3.10-cuda11.8-cudnn8-transformer4.30.0:v1.0.0-20240419-173534-a40fd64e1' \
    --storage-mount 53c47ef7-3457-11ef-ae95-ee12bf38b649:/mnt/afs \
    --training-framework=pytorch \
    --worker-nodes=1 \
    --worker-spec='N10lP.nn.A80a.1' \
    --command="ln -s /mnt/afs/yangzekang/envs/modelgen /usr/local/lib/miniconda3/envs/modelgen
source activate modelgen
cd /mnt/afs/yangzekang/ModelGen
python train_darts_cifar.py --model_name {{model_name}} --auxiliary --cutout --code-dir {{code_dir}} --output {{train_log_dir}} --epochs 100"
"""

TRAIN_DARTS_CIFAR10_LAYER20_4090D = f"""sco acp jobs create --workspace-name=tetras-share \
    --aec2-name=tetrasvideo4090d \
    --job-name={{job_name}} \
    --container-image-url='registry.cn-sh-01.sensecore.cn/studio-aicl/ubuntu20.04-py3.10-cuda11.8-cudnn8-transformer4.30.0:v1.0.0-20240419-173534-a40fd64e1' \
    --storage-mount 53c47ef7-3457-11ef-ae95-ee12bf38b649:/mnt/afs \
    --training-framework=pytorch \
    --worker-nodes=1 \
    --worker-spec='N10lP.nn.A80a.1' \
    --command="ln -s /mnt/afs/yangzekang/envs/modelgen /usr/local/lib/miniconda3/envs/modelgen
source activate modelgen
cd /mnt/afs/yangzekang/ModelGen
python train_darts_cifar.py --model_name {{model_name}} --auxiliary --cutout --code-dir {{code_dir}} --output {{train_log_dir}} --tag full_cifar10 --drop-path-prob 0.2"
"""

TRAIN_DARTS_CIFAR100_LAYER20_4090D = f"""sco acp jobs create --workspace-name=tetras-share \
    --aec2-name=tetrasvideo4090d \
    --job-name={{job_name}}_cifar100 \
    --container-image-url='registry.cn-sh-01.sensecore.cn/studio-aicl/ubuntu20.04-py3.10-cuda11.8-cudnn8-transformer4.30.0:v1.0.0-20240419-173534-a40fd64e1' \
    --storage-mount 53c47ef7-3457-11ef-ae95-ee12bf38b649:/mnt/afs \
    --training-framework=pytorch \
    --worker-nodes=1 \
    --worker-spec='N10lP.nn.A80a.1' \
    --command="ln -s /mnt/afs/yangzekang/envs/modelgen /usr/local/lib/miniconda3/envs/modelgen
source activate modelgen
cd /mnt/afs/yangzekang/ModelGen
python train_darts_cifar.py --model_name {{model_name}} --auxiliary --cutout --code-dir {{code_dir}} --output {{train_log_dir}} --tag full_cifar100 --num-classes 100 --drop-path-prob 0.2 --weight-decay 0.0005"
"""

NB201_TRAIN_TEMPLATE_MAP = {
    'l40':{
        'imagenet-1k':TRAIN_IMAGENET_l40,
        'nas-bench-201-cifar10':TRAIN_NAS_BENCH_201_CIFAR10_l40,
        'nas-bench-201-cifar100':TRAIN_NAS_BENCH_201_CIFAR100_l40,
        'nas-bench-201-imagenet16-120':TRAIN_NAS_BENCH_201_IMAGENET16_120_l40
    },
    '4090d':{
        'nas-bench-201-cifar10':TRAIN_NAS_BENCH_201_CIFAR10_4090D,
        'nas-bench-201-cifar100':TRAIN_NAS_BENCH_201_CIFAR100_4090D,
        'nas-bench-201-imagenet16-120':TRAIN_NAS_BENCH_201_IMAGENET16_120_4090D
    }
}




