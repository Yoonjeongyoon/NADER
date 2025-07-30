from functools import wraps
from time import sleep
import torch
import numpy as np
import random
import json
import re
import os


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_json_list(path):
    if os.path.isfile(path):
        with open(path,'r') as f:
            ds = json.load(f)
    else:
        ds = []
    return ds

def load_json_dict(path):
    if os.path.isfile(path):
        with open(path,'r') as f:
            ds = json.load(f)
    else:
        ds = {}
    return ds

def save_json(anno,path):
    with open(path,'w') as f:
        json.dump(anno,f,indent='\t')


def retry(retries: int = 3, delay: float = 1):
    """
    函数执行失败时，重试

    :param retries: 最大重试的次数
    :param delay: 每次重试的间隔时间，单位 秒
    :return:
    """

    # 校验重试的参数，参数值不正确时使用默认参数
    if retries < 1 or delay <= 0:
        retries = 3
        delay = 1

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 第一次正常执行不算重试次数，所以retries+1
            for i in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # 检查重试次数
                    if i == retries:
                        print(f"Error: {repr(e)}")
                        print(f'"{func.__name__}()" 执行失败，已重试{retries}次')
                        break
                    else:
                        print(
                            f"Error: {repr(e)}，{delay}秒后第[{i+1}/{retries}]次重试..."
                        )
                        sleep(delay)

        return wrapper

    return decorator


def load_block(path):
    ret = []
    with open(path,'r') as f:
        txt = f.read()
        s = re.findall('(##.*?##(.(?!##))*)',txt,re.DOTALL)
        for x in s:
            ret.append(x[0].strip())
    return ret
