import os

class Log:

    def __init__(self,root_dir):
        super().__init__()
        self.root_dir = root_dir
        os.makedirs(root_dir,exist_ok=True)
    
    def update(self,name='val_res',epoch=0,acc1=0,acc5=0):
        path = os.path.join(self.root_dir,f'{name}.txt')
        if not os.path.isfile(path):
            with open(path,'w') as f:
                f.write('epoch,acc1,acc5\n')
        with open(path,'a') as f:
            f.write(f'{epoch},{acc1},{acc5}\n')