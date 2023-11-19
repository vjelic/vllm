import torch
import torch.nn.functional as F
from rocsolidxgemm import rocb_create_extension,rocb_mm
from hipbsolidxgemm import hipb_create_extension,hipb_mm
import os
import yaml
import pandas as pd
from vllm import custom_ops


class TunedGemm:
    def __init__(self):
        #rocb_create_extension()
        #hipb_create_extension()
        self.extensions_created = False
        self.bestsols = {}
        self.load_best_sols()
        self.create_ds()
    def load_best_sols(self):
        perfbits = {}
        perf_file = os.environ.get('VLLM_PERF_YAML')
        if perf_file is not None:
            with open(perf_file, 'r') as file:
                perfbits = yaml.safe_load(file)
        if torch.distributed.get_rank() == 0:
            print('>>>Importing Tuned Gemm Solutions',perf_file, perfbits)
        tune_file = perfbits.get('tuned_gemm_csv',None)
        if tune_file is not None:
            self.bestsols = pd.read_csv(tune_file,index_col=[0])
            print(self.bestsols)
    def create_ds(self):
        df = self.bestsols
        solds = {}
        for i in range(len(df)):
            ds = df.iloc[i]
            key = (ds['M'],ds['N'],ds['K'])
            if ds['libtype']=='hipblaslt': soltype = 1
            elif ds['libtype']=='rocblas': soltype = 2
            elif ds['libtype']=='custom': soltype = 3
            solds[key] = (soltype,int(ds['solidx']))
        self.solids = solds
    def query_sol(self,m,n,k):
        return self.solids.get((m,n,k),(0,0))
    def mm(self,inp,weights):
        if self.extensions_created == False:
            rocb_create_extension()
            hipb_create_extension()
            self.extensions_created = True
        soltype,solidx = self.query_sol(m=weights.shape[0],n=inp.shape[0],k=inp.shape[1])
        if soltype==1:
            out = hipb_mm(inp,weights.t(),solidx)
        elif soltype==3:
            ##only matvec is supported currently
            out = torch.empty(inp.shape[0],weights.shape[0],dtype=torch.float16,device='cuda')
            if solidx<=1:
                custom_ops.LLMM1(weights,inp,out)
        elif soltype==2:
            out = rocb_mm(inp,weights.t(),solidx)
        else:
            #print('>>>Tgemm Default',inp.shape,weights.shape,soltype,solidx)
            out = F.linear(inp,weights)
        return out

tgemm = TunedGemm()
