from train import mytrain
import sys
from contextlib import redirect_stdout, redirect_stderr
if __name__=='__main__':
    #gmf/mlp
    # name_map={0:'gmf',1:'mlp',2:'neumf'}
    # for model in [0,1]:
    #     for dim in [8,16,32,64]:
    #         log_path=f'train_{name_map[model]}_dim{dim}.log'
    #         with open(log_path,'w',encoding='utf-8') as f:
    #           with redirect_stdout(f), redirect_stderr(f):
    #            mytrain(dim,model,1)
    #neumf
    log_path='train_neumf_dim64.log'
    with open(log_path,'w',encoding='utf-8') as f:
     with redirect_stdout(f), redirect_stderr(f):
        mytrain(64,2,1)