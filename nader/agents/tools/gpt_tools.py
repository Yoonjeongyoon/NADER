
def convert_ChatCompeletion2json(response):
    response = dict(response)
    for i,choice in enumerate(response['choices']):
        choice = dict(choice)
        choice['message'] = dict(choice['message'])
        response['choices'][i] = choice
    response['usage'] = dict(response['usage'])
    return response
    
def calculate_money(json,prompt_price=0.03,res_price=0.06):
    prompt_tokens = json['usage']['prompt_tokens']
    completion_tokens = json['usage']['completion_tokens']
    price = prompt_tokens/1000*prompt_price+completion_tokens/1000*res_price
    return price,price*7.3