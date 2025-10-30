TRAIN_BATCH = """directory="{task_dir}"
models={models}

cd "{task_dir}"
echo "DEBUG: SEQUENTIAL"
for file in "${{models[@]}}"; do
  [[ "$file" != *.sh ]] && continue
  echo "DEBUG: run $file"
  # 로그를 파일별로 저장 (교착 방지)
  bash "$file" >> "$file.log" 2>&1
  rc=$?
  echo "DEBUG: $file rc=$rc"
  # Continue even if one model fails (changed for robust training)
  # if [[ $rc -ne 0 ]]; then exit $rc; fi
done
"""


TRAIN_DETECTION_COCO2017_LOCAL = """echo "DEBUG: Starting detection training for model {model_name}"
echo "DEBUG: Current directory: $(pwd)"
cd {nader_root}

# Create model-specific config
MODEL_CONFIG_DIR="{train_log_dir}/{model_name}/1"
mkdir -p "$MODEL_CONFIG_DIR"

# Get absolute path for MODEL_CONFIG_DIR
MODEL_CONFIG_DIR=$(realpath "$MODEL_CONFIG_DIR")
echo "DEBUG: MODEL_CONFIG_DIR absolute path: $MODEL_CONFIG_DIR"

# Get absolute path for code_dir blocks
BLOCKS_DIR=$(realpath "{code_dir}/blocks")
echo "DEBUG: BLOCKS_DIR absolute path: $BLOCKS_DIR"

# Generate model-specific config file
cat > "$MODEL_CONFIG_DIR/retinanet_r50_nader_fpn_1x_coco.py" << EOF
_base_ = [
    '{nader_root}/mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py',
]

model = dict(
    neck=dict(
        type='NADERFPNAdapter',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        start_level=1,
        add_extra_convs='on_input',
        block_prefix='{model_name}',
        blocks_dir='$BLOCKS_DIR'
    )
)
EOF

echo "DEBUG: Running MMDetection training"
cd {nader_root}/mmdetection
python tools/train.py "$MODEL_CONFIG_DIR/retinanet_r50_nader_fpn_1x_coco.py" \\
    --work-dir "$MODEL_CONFIG_DIR/work_dir"
TRAIN_EXIT_CODE=$?

# Trap to ensure status file is always written (even on kill/interrupt)
trap 'echo "interrupted" > "$MODEL_CONFIG_DIR/train_status.txt"; exit 1' INT TERM

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "DEBUG: Training failed with exit code $TRAIN_EXIT_CODE"
    echo "failed" > "$MODEL_CONFIG_DIR/train_status.txt"
    echo "0.0" > "$MODEL_CONFIG_DIR/val_acc.txt"
    exit 0
fi

echo "DEBUG: Extracting mAP from training log"
# Extract mAP from training log with improved error handling
python -c "
import re
import glob
import sys
import os

model_config_dir = '$MODEL_CONFIG_DIR'
try:
    # Parse from log file
    log_pattern = model_config_dir + '/work_dir/*/*.log'
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        # Try alternative pattern
        log_pattern = model_config_dir + '/work_dir/*.log'
        log_files = glob.glob(log_pattern)
    
    if log_files:
        log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        with open(log_files[0], 'r') as f:
            log_content = f.read()
        
        # Find the last mAP result from validation
        mAP_matches = re.findall(r'coco/bbox_mAP:\s*([\d.]+)', log_content)
        if mAP_matches:
            mAP = float(mAP_matches[-1])
            print(f'Found mAP: {{mAP}}')
        else:
            print('WARNING: mAP not found in log, using 0.0')
            mAP = 0.0
    else:
        print(f'WARNING: No log file found matching {{log_pattern}}')
        mAP = 0.0
except Exception as e:
    print(f'ERROR extracting mAP: {{e}}')
    mAP = 0.0
    sys.exit(1)

# Write mAP to val_acc.txt
try:
    with open(model_config_dir + '/val_acc.txt', 'w') as f:
        f.write(str(mAP))
    print(f'Successfully wrote mAP ({{mAP}}) to val_acc.txt')
except Exception as e:
    print(f'ERROR writing val_acc.txt: {{e}}')
    sys.exit(1)
" 
PARSE_EXIT_CODE=$?

# Always write status file based on parsing result
if [ $PARSE_EXIT_CODE -eq 0 ]; then
    echo "DEBUG: Detection training completed successfully"
    echo "done" > "$MODEL_CONFIG_DIR/train_status.txt"
else
    echo "DEBUG: mAP parsing failed, marking as failed"
    echo "failed" > "$MODEL_CONFIG_DIR/train_status.txt"
fi

# Final check: ensure status file was created
if [ ! -f "$MODEL_CONFIG_DIR/train_status.txt" ]; then
    echo "ERROR: train_status.txt was not created! Force writing..."
    echo "unknown" > "$MODEL_CONFIG_DIR/train_status.txt"
fi

echo "DEBUG: Final status check - $(cat $MODEL_CONFIG_DIR/train_status.txt)"
"""
DETECTION_TRAIN_TEMPLATE_MAP_LOCAL = {
    'detection-bench-coco2017': TRAIN_DETECTION_COCO2017_LOCAL,
}   
TRAIN_NAS_BENCH_201_CIFAR10_LOCAL = """echo "DEBUG: Starting training script for model {model_name}"
echo "DEBUG: Current directory: $(pwd)"
cd {nader_root}

echo "DEBUG: Running train_cifar10.py"
python train_cifar10.py \
    --model_name {model_name} \
    --tag 1 \
    --code-dir {code_dir} \
    --output {train_log_dir}

echo "DEBUG: train_cifar10.py completed with exit code: $?"
echo "done" > "{train_log_dir}/{model_name}/1/train_status.txt"
"""

TRAIN_NAS_BENCH_201_CIFAR100_LOCAL = """cd {nader_root}
python train_cifar100.py \
    --model_name {model_name} \
    --tag 1 \
    --code-dir {code_dir} \
    --output {train_log_dir}
echo "done" > "{train_log_dir}/{model_name}/1/train_status.txt"
"""

TRAIN_NAS_BENCH_201_IMAGENET16_120_LOCAL = """cd {nader_root}
python train_imagenet16_120.py \
    --model_name {model_name} \
    --tag 1 \
    --code-dir {code_dir} \
    --output {train_log_dir}
echo "done" > "{train_log_dir}/{model_name}/1/train_status.txt"
"""

NB201_TRAIN_TEMPLATE_MAP_LOCAL = {
    'nas-bench-201-cifar10': TRAIN_NAS_BENCH_201_CIFAR10_LOCAL,
    'nas-bench-201-cifar100': TRAIN_NAS_BENCH_201_CIFAR100_LOCAL,
    'nas-bench-201-imagenet16-120': TRAIN_NAS_BENCH_201_IMAGENET16_120_LOCAL,
}

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
    },
    'local': NB201_TRAIN_TEMPLATE_MAP_LOCAL
}

DETECTION_TRAIN_TEMPLATE_MAP = {
    'local': DETECTION_TRAIN_TEMPLATE_MAP_LOCAL,
}



