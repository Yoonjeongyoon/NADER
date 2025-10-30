import time
import os

from openai import OpenAI

def open_proxy():
    os.environ["HTTP_PROXY"] = "http://0.0.0.0:7890"
    os.environ["HTTPS_PROXY"] = "http://0.0.0.0:7890"

def close_proxy():
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)


def call_llm(messages,max_try=50,use_proxy=False,*agrs,**kwds):
    model_name = os.environ['LLM_MODEL_NAME']
    if isinstance(messages,str):
        messages = [{"role":"user","content":messages}]
    if use_proxy:
        open_proxy()
    if 'deepseek' in model_name:
        client = OpenAI(api_key=os.environ['API_KEY_DEEPSEEK'], base_url="https://api.deepseek.com", max_retries=10)
    elif 'gpt' in model_name.lower():
        client = OpenAI(max_retries=10)
    else:
        raise NotImplementedError
    for i in range(max_try):
        try:
            response = client.chat.completions.create(model=model_name, messages=messages)
            break
        except Exception as e:
            time.sleep(3)
            if i!=max_try-1:
                print(str(e)+'retry')
                continue
            elif (i+1)%10:
                time.sleep(3)
            raise
    if use_proxy:
        close_proxy()
    response = {
        "prompt_tokens":response.usage.prompt_tokens,
        "completion_tokens":response.usage.completion_tokens,
        'model':response.model,
        "content":response.choices[0].message.content
    }
    return response

if __name__=='__main__':
    for i in range(1):
        message = [{"role": "user", "content": """鲁迅为什么暴打周树人"""}]
        res = call_llm(message)
        print(res)