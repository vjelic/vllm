import torch
import os
import argparse
import rocsolidxgemm
import hipbsolidxgemm
import numpy as np
import torch.nn.functional as F
import sys
import pandas as pd
import json
import random
rocsolidxgemm.rocb_create_extension()
hipbsolidxgemm.hipb_create_extension()

rtol = 1e-5
atol = 1
dtype = torch.float16

class Gemm:
    def __init__(self,m,n,k,dtype,rocblas_decode=False):
        self.m=m
        self.k=k
        self.n=n
        self.dtype=dtype
        self.nb = 37
        self.inp = torch.randn((self.n, self.k), dtype=self.dtype, device='cuda')
        self.weights = torch.randn((self.m, self.k), dtype=self.dtype, device='cuda')
        #weights2 is used in measurement/warm iters to ensure HBM fetch for weight tensors
        self.weights2 = torch.randn((self.nb, self.m, self.k), dtype=self.dtype, device='cuda')
        self.blob = torch.ones(128*1024*1024,dtype=torch.float32,device='cuda')
        self.topn = 20 #number of top solutions from each source
        self.hipb_sols=[]
        self.rtol = 1e-5
        self.atol = 1
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.hipb_prefer_ratio = 0.995 #prefer hipblaslt unless rocblas time is less than this ratio of hipblaslt time
        self.rocblas_decode=rocblas_decode
    def find_hipblas_sols(self):
        sols = hipbsolidxgemm.hipb_findallsols(self.inp,self.weights.t())
        print('M N K',self.m,self.n,self.k,'>>> Total hipb solutions',len(sols), flush=True)
        #print(sols)
        self.hipb_sols = sols
    def check_gemm_ref(self,libtype,solidx):
        ref = F.linear(self.inp,self.weights)
        if libtype == 'hipblaslt':
            c = hipbsolidxgemm.hipb_mm(self.inp,self.weights.t(),solidx)
        elif libtype == 'rocblas':
            c = rocsolidxgemm.rocb_mm(self.inp,self.weights.t(),solidx)
        if torch.allclose(c, ref, atol=self.atol,  rtol=self.rtol):
            #print('>>>',libtype,'Solidx',solidx,'passed reference test')
            return True
        else:
            print('>>>',libtype,'Solidx',solidx,'FAILED reference test', flush=True)
            print(ref, flush=True)
            print(c, flush=True)
            return False
    def hipb_time_sol(self,solidx,cold_iters=2,warm_iters=10):
        #print('>>>hipbtime',solidx)
        for i in range(cold_iters):
            c = hipbsolidxgemm.hipb_mm(self.inp,self.weights.t(),solidx)
        self.start.record()
        for i in range(warm_iters):
            c = hipbsolidxgemm.hipb_mm(self.inp,self.weights2[random.randint(0,self.nb-1)].t(),solidx)
        self.end.record()
        torch.cuda.synchronize()
        gtime = self.start.elapsed_time(self.end)/warm_iters
        #print('>>> Solidx GTime',solidx,gtime,'ms')
        return gtime
    def hipb_time_all_sols(self,fast_mode=0,top_sols=0):
        coldi=20; warmi=20
        if fast_mode: coldi=2; warmi=2
        solutions = self.hipb_sols
        if top_sols: solutions = self.hipb_top_sols
        gtimes = {}
        for solidx in solutions:
            gtimes[solidx] = self.hipb_time_sol(solidx,cold_iters=coldi,warm_iters=warmi)
        self.hipb_gtimedf = pd.DataFrame.from_dict(gtimes,orient='index',columns=['gtimems']).sort_values(by='gtimems')
        self.hipb_gtimedf.to_csv('/tmp/hipb_gtimedf.csv')
        print('>>> HipBlasLt top solutions, Fast Mode',fast_mode)
        print(self.hipb_gtimedf.head(self.topn))
    def rocb_time_sol(self,solidx,cold_iters=2,warm_iters=10):
        for i in range(cold_iters):
            c = rocsolidxgemm.rocb_mm(self.inp,self.weights.t(),solidx)
        self.start.record()
        for i in range(warm_iters):
            c = rocsolidxgemm.rocb_mm(self.inp,self.weights2[random.randint(0,self.nb-1)].t(),solidx)
        self.end.record()
        torch.cuda.synchronize()
        gtime = self.start.elapsed_time(self.end)/warm_iters
        #print('>>> RocSolidx GTime',solidx,gtime,'ms')
        return gtime
    def find_rocblas_sols(self):
        sols = rocsolidxgemm.rocb_findallsols(self.inp,self.weights.t())
        print('M N K',self.m,self.n,self.k,'>>> Total rocb solutions',len(sols), flush=True)
        #print(sols)
        self.rocb_sols = sols
    def rocb_time_all_sols(self,fast_mode=0,top_sols=0):
        coldi=20; warmi=20
        if fast_mode: coldi=2; warmi=2
        solutions = self.rocb_sols
        if top_sols: solutions = self.rocb_top_sols
        gtimes = {}
        for solidx in solutions:
            gtimes[solidx] = self.rocb_time_sol(solidx,coldi,warmi)
        self.rocb_gtimedf = pd.DataFrame.from_dict(gtimes,orient='index',columns=['gtimems']).sort_values(by='gtimems')
        self.rocb_gtimedf.to_csv('/tmp/rocb_gtimedf.csv')
        print('>>> Rocblas top solutions, Fast Mode',fast_mode, flush=True)
        print(self.rocb_gtimedf.head(self.topn), flush=True)
    def warmup(self,warmi=500):
        for i in range(warmi):
            self.blob = self.blob + 0.00001
    def functional_check_topn_fastest(self):
        rocb_topn = []
        for solidx in self.rocb_gtimedf.index[:self.topn]:
            if self.check_gemm_ref(libtype='rocblas',solidx=solidx):
                rocb_topn.append(solidx)
        self.rocb_top_sols = rocb_topn
        hipb_topn = [] 
        for solidx in self.hipb_gtimedf.index[:self.topn]:
            if self.check_gemm_ref(libtype='hipblaslt',solidx=solidx):
                hipb_topn.append(solidx)
        self.hipb_top_sols = hipb_topn

    def find_fastest_solution(self):
        self.find_rocblas_sols()
        if not (self.rocblas_decode and self.n == 1):
            self.find_hipblas_sols()
        self.warmup()
        self.rocb_time_all_sols(fast_mode=1)
        self.warmup()
        self.hipb_time_all_sols(fast_mode=1)
        self.functional_check_topn_fastest()
        self.warmup()
        self.rocb_time_all_sols(fast_mode=0,top_sols=1)
        self.warmup()
        self.hipb_time_all_sols(fast_mode=0,top_sols=1)
        if len(self.rocb_gtimedf)>0 and len(self.hipb_gtimedf)>0:
            best_rocb_time = self.rocb_gtimedf.gtimems.iloc[0]
            best_hipb_time = self.hipb_gtimedf.gtimems.iloc[0]
            if best_rocb_time<best_hipb_time*self.hipb_prefer_ratio:
                self.best_libtype = 'rocblas'
                self.best_solidx = self.rocb_gtimedf.index[0]
                self.best_soltime = best_rocb_time
            else:
                self.best_libtype = 'hipblaslt'
                self.best_solidx = self.hipb_gtimedf.index[0]
                self.best_soltime = best_hipb_time
            #self.check_gemm_ref(self.best_libtype,self.best_solidx)
        elif len(self.hipb_gtimedf)>0:
                print('>>> Only hipblas solutions found!',flush=True)
                best_hipb_time = self.hipb_gtimedf.gtimems.iloc[0]
                self.best_libtype = 'hipblaslt'
                self.best_solidx = self.hipb_gtimedf.index[0]
                self.best_soltime = best_hipb_time
        elif len(self.rocb_gtimedf)>0:
                print('>>> Only rocblas solutions found!',flush=True)
                best_rocb_time = self.rocb_gtimedf.gtimems.iloc[0]
                self.best_libtype = 'rocblas'
                self.best_solidx = self.rocb_gtimedf.index[0]
                self.best_soltime = best_rocb_time
        else:
            print('>>> No rocblas or hipblas solutions found!',flush=True)
            self.best_libtype = 'rocblas'
            self.best_solidx = 0
            self.best_soltime = 0
        print('>>> Fastest Solution is',self.best_libtype,self.best_solidx,self.best_soltime,flush=True)


class GemmTuner:
    def __init__(self, dtype, rocblas_decode=False):
        self.gemm_problems = pd.DataFrame(columns=['M','N','K'])
        self.dtype = dtype
        self.rocblas_decode = rocblas_decode
    def add_gemm(self,m,n,k):
        entry = {'M': [m], 'N' : [n], 'K' : [k]}
        df = pd.DataFrame(entry)
        self.gemm_problems = pd.concat([self.gemm_problems,df],ignore_index=True)
    def find_best_sols(self,outfile):
        df = self.gemm_problems
        soldf = pd.DataFrame()
        for i in range(len(df)):
            ds = df.iloc[i]
            gemmobj = Gemm(ds['M'],ds['N'],ds['K'],dtype=self.dtype, rocblas_decode=self.rocblas_decode)
            gemmobj.find_fastest_solution()
            soldf.loc[i,'libtype'] = gemmobj.best_libtype
            soldf.loc[i,'solidx'] = gemmobj.best_solidx
            soldf.loc[i,'soltimems'] = gemmobj.best_soltime
        soldf['dtype'] = self.dtype
        self.soldf = soldf
        finaldf = pd.concat([self.gemm_problems,self.soldf],axis=1)
        finaldf.to_csv(outfile)
        print(finaldf)

'''
{'architectures': ['LlamaForCausalLM'], 'bos_token_id': 1, 'eos_token_id': 2, 'hidden_act': 'silu', 'hidden_size': 5120, 'initializer_range': 0.02, 
'intermediate_size': 13824, 'max_position_embeddings': 2048, 'model_type': 'llama', 'num_attention_heads': 40, 'num_hidden_layers': 40, 'num_key_value_heads': 40, 
'pretraining_tp': 1, 'rms_norm_eps': 1e-05, 'rope_scaling': None, 'tie_word_embeddings': False, 'torch_dtype': 'float16', 'transformers_version': '4.33.0.dev0', 'use_cache': True, 'vocab_size': 32000}
'''
def generate_mk_sets(model_dir, tp=1):
    f = open(f'{model_dir}/config.json')
    data = json.load(f)
    hidden_size = data['hidden_size']
    intermediate_size = data['intermediate_size']
    total_num_heads = data['num_attention_heads']
    total_num_kv_heads = data['num_key_value_heads']
    head_dim = hidden_size // total_num_heads


    return [((total_num_heads + (2*total_num_kv_heads)) * head_dim // tp, hidden_size), (hidden_size, hidden_size // tp), (intermediate_size *2 // tp, hidden_size), (hidden_size, intermediate_size // tp) ], hidden_size

def get_dtype(dtype_str):
    dtype = torch.float16
    if dtype_str == 'f32':
        dtype = torch.float32
    elif dtype_str == 'bf16':
        dtype = torch.bfloat16
    elif dtype_str == 'f16':
        dtype = torch.float16
    else:
        print('>>> Warning! Invalid dtype', dtype_str, 'using default dtype f16')
    return dtype


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.getenv('GTUNE_MODEL', ""), help="Enter the location of your model directory")
    parser.add_argument("--output", type=str, default=os.getenv('GTUNE_OUTPUT', "tuned.csv"), help="output file for tuned gemm solutions")
    parser.add_argument("--tp", type=int, default=os.getenv('GTUNE_TP', 1), help="Tensor parallelism to be used.")
    parser.add_argument("--dtype", type=str, default='f16', help="dtype f32 f16 bf16")
    parser.add_argument("--rocblas-decode", action="store_true", default=False, help="forces rocblas solution on decode N=1")
    args = parser.parse_args()

    dtype = get_dtype(args.dtype)

    gtuner = GemmTuner(dtype, args.rocblas_decode)
    nsets = [1,512,1024,2048,3072,4096,6144,8192,16384]
    if not args.model_dir:
        print(">>> Warning! NO MODEL SPECIFIED. Tuning for LL2 13B TP1")
        #LL2 13B sizes
        mksets = [(15360,5120),(5120,5120),(27648,5120),(5120,13824)]
        gtuner.add_gemm(m=32000,n=1,k=5120) #logits gemm
    else:
        mksets, hidden_size = generate_mk_sets(args.model_dir, args.tp)
        gtuner.add_gemm(m=32000//args.tp,n=1,k=hidden_size) #TODO: Handle cases where vocab_size is not divisible by tp

    for n in sorted(nsets):
        for m,k in mksets:
            gtuner.add_gemm(m,n,k)

    gtuner.find_best_sols(args.output)

