# Copyright 2023 The vLLM team.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import os

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_group_half_pair,
)
from .mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)

from .random import get_cuda_rng_tracker
from .utils import (
    divide,
    VocabUtility,
)

from torch._C._distributed_c10d import (
    AllreduceOptions,
    ReduceOp
)
from vllm.model_executor.layers.tuned_gemm import tgemm

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False,
                                      'partition_dim': -1,
                                      'partition_stride': 1}

def param_is_not_tensor_parallel_duplicate(param):
    return (hasattr(param, 'tensor_model_parallel') and
            param.tensor_model_parallel) or (
                get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute,
                    getattr(source_tensor, attribute))
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False,
                                  *, params_dtype=None):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    if params_dtype is None:
        params_dtype = torch.get_default_dtype()

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    master_weight = master_weight.to(dtype=params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None

def splitk_linear(inp,w,splitk=2):
    wsp = torch.chunk(w,splitk,dim=1)
    isp = torch.chunk(inp,splitk,dim=1)
    #print('>>>',isp[0].shape,wsp[1].shape)
    cnew = []
    for i in range(splitk):
        cnew.append(F.linear(isp[i],wsp[i]))
    c = cnew[0]
    for i in range(1,splitk): c.add_(cnew[i])
    return c

def splitm_linear(inp,w,splitm=2,splits=None,splitk=1):
    outputp=[]
    if splits is not None:
        wsp = torch.split(w,splits)
    else:
        wsp = torch.chunk(w,splitm)
    for i,_ in enumerate(wsp): 
        #print('>>>wspi',wsp[i].shape)
        if splitk==1:
            outputp.append(F.linear(inp, wsp[i]))
        else:
            outputp.append(splitk_linear(inp,wsp[i],splitk))
    return torch.cat((outputp),dim=1) 


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.

    Keyword Arguments:
        init_method: method to initialize weights.
        params_dtype
        use_cpu_initialization
        perform_initialization
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, *,
                 init_method=init.xavier_normal_,
                 params_dtype: torch.dtype=None,
                 use_cpu_initialization: bool=False,
                 perform_initialization: bool=True):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # Set the defaults for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights and initialize.
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight, self.num_embeddings, self.embedding_dim,
                    self.num_embeddings_per_partition, 0, init_method,
                    params_dtype=params_dtype)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method,
                                              partition_dim=0, stride=1)

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
        params_dtype:
        use_cpu_initialization:
    """

    def __init__(self, input_size, output_size, *,
                 bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False,
                 params_dtype=None,
                 use_cpu_initialization=False,
                 perform_initialization=True,
                 ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        self.world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, self.world_size)
        self.skip_bias_add = skip_bias_add

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=params_dtype))
            if perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight, self.output_size, self.input_size,
                    self.output_size_per_partition, 0, init_method,
                    stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                device=torch.cuda.current_device(), dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method,
                                              partition_dim=0, stride=stride)

        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype))
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        #global gradlib_handle_created
        #if gradlib_handle_created == False:
        #    create_extension()
        #    gradlib_handle_created = True
        #    print('>>> create gradlib extension')
        #global rocsolidxgemm_handle_created
        #if rocsolidxgemm_handle_created == False:
        #    rocb_create_extension()
        #    rocsolidxgemm_handle_created = True
        #    print('>>> create rocsolidxgemm extension')


    def forward(self, input_,call_hipblaslt=0,splitm=1,splitk=1,splits=None,call_rocsolidx=0):
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        bias = self.bias if not self.skip_bias_add else None

        input_parallel = input_
        # Matrix multiply.
        #print('>>>col',input_parallel.shape, self.weight.shape)
        output_parallel = tgemm.mm(input_, self.weight)
        #if call_hipblaslt:
        #    output_parallel = hipb_mm(input_, self.weight.t(),call_hipblaslt)
        #elif call_rocsolidx:
        #    output_parallel = rocb_mm(input_, self.weight.t(),call_rocsolidx)
        #elif splitm>1:
        #    output_parallel = splitm_linear(input_parallel,self.weight,splitm=splitm,splits=splits,splitk=splitk)
        #elif splitk>1:
        #    output_parallel = splitk_linear(input_parallel,self.weight,splitk)
        #else:
        #    output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments:
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
        params_dtype:
        use_cpu_initialization:
        perform_initialization:
        reduce_results:
    """

    def __init__(self, input_size, output_size, *,
                 bias=True, input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False,
                 params_dtype=None,
                 use_cpu_initialization=False,
                 perform_initialization=True,
                 reduce_results=True,
                 ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # Divide the weight matrix along the last dimension.
        self.world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, self.world_size)
        self.skip_bias_add = skip_bias_add

        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the "
                             "results can lead to incorrect results")

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size_per_partition,
                                                dtype=params_dtype))
            if perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight, self.output_size, self.input_size,
                    self.input_size_per_partition, 1, init_method,
                    stride=stride, return_master_weight=keep_master_weight_for_test,
                    params_dtype=params_dtype)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                device=torch.cuda.current_device(), dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method,
                                              partition_dim=1, stride=stride)
        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                                                  dtype=params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=params_dtype))

            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.weight_t = self.weight.t()
        if self.world_size>1:
            self.ropts = AllreduceOptions()
            self.ropts.reduceOp = ReduceOp.SUM
            self.group=get_tensor_model_parallel_group()
            self.group_half,self.group_pair=get_tensor_model_parallel_group_half_pair()
            self.fastreduce = FastReduce()

        #global gradlib_handle_created
        #if gradlib_handle_created == False:
        #    create_extension()
        #    gradlib_handle_created = True
        #    print('>>> create gradlib extension')
        #global rocsolidxgemm_handle_created
        #if rocsolidxgemm_handle_created == False:
        #    rocb_create_extension()
        #    rocsolidxgemm_handle_created = True
        #    print('>>> create rocsolidxgemm extension')

        self.graphs = {}

    def forward(self, input_, call_hipblaslt=0, splitm=1, splitk=1, splits=None, call_rocsolidx=0, graphx=0):
        if graphx:
            n = input_.shape[0]
            #if n<16:
            if n not in self.graphs.keys():
                    print('>>> Warm up and Graph n',n)
                    inpx = torch.ones_like(input_)
                    for i in range(3):
                        outx, _ = self.dyn_forward(inpx, call_hipblaslt, splitm, splitk, splits, call_rocsolidx)
                    #torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    #print(">>> Capturing graph!")
                    with torch.cuda.graph(g):
                        #torch.cuda.current_stream().wait_stream(torch.cuda.default_stream())
                        outx, _ = self.dyn_forward(inpx, call_hipblaslt, splitm, splitk, splits, call_rocsolidx)
                    #torch.cuda.synchronize()
                    self.graphs[n] = (g,inpx,outx)
            else:
                    #print(">>> Replaying graph!",n)
                    (g,inpx,outx) = self.graphs[n]
                    inpx.copy_(input_)
                    g.replay()
                    return outx,None

        return self.dyn_forward(input_, call_hipblaslt, splitm, splitk, splits, call_rocsolidx)

    def dyn_forward(self, input_, call_hipblaslt=0, splitm=1, splitk=1, splits=None, call_rocsolidx=0):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        #print('>>>row',input_parallel.shape, self.weight.shape)
        output_parallel = tgemm.mm(input_parallel, self.weight)
        #if call_hipblaslt:
        #    output_parallel = hipb_mm(input_parallel, self.weight.t(),call_hipblaslt)
        #elif call_rocsolidx:
        #    output_parallel = rocb_mm(input_parallel, self.weight.t(),call_rocsolidx)
        #elif splitm>1:
        #    output_parallel = splitm_linear(input_parallel,self.weight,splitm=splitm,splits=splits)
        #elif splitk>1:
        #    output_parallel = splitk_linear(input_parallel,self.weight,splitk)
        #else:
        #    output_parallel = F.linear(input_parallel, self.weight)
        #output_parallel = F.linear(input_parallel, self.weight)
        if self.reduce_results and self.world_size > 1:
            #output_ = reduce_from_tensor_model_parallel_region(output_parallel)
            #torch.distributed.all_reduce(output_parallel, group=get_tensor_model_parallel_group())
            work = self.group.allreduce([output_parallel], self.ropts)
            work.wait()
            #output_parallel = self.fastreduce(output_parallel)

            #output_parallel2 = torch.clone(output_parallel)
            #work = self.group_half.allreduce([output_parallel], self.ropts)
            #work.wait()
            #torch.distributed.barrier()
            #work = self.group_pair.allreduce([output_parallel], self.ropts)
            #work.wait()
            output_ = output_parallel
        else:
            output_ = output_parallel

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

shape_dict = {}
graph_dict = {}

class FastReduce(torch.nn.Module):
    def __init__(self):
        super(FastReduce, self).__init__()
        self.ropts = AllreduceOptions()
        self.ropts.reduceOp = ReduceOp.SUM
        self.group=get_tensor_model_parallel_group()
        #self.shape_dict = {}
        self.graph_bs = int(os.environ.get('VLLM_GRAPH_ALLREDUCE_BS',0))
    
    def warmup_and_graph(self,x):
        print('>>>warmup nccl',x.shape)
        inputx = x.clone()
        for i in range(1):
            work = self.group.allreduce([inputx], self.ropts)
            work.wait()

        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        #s = torch.cuda.Stream() 
        print(">>> Capturing graph!",inputx.shape) 
        with torch.cuda.graph(g):
            work = self.group.allreduce([inputx], self.ropts)
            work.wait()

        #torch.cuda.synchronize()
        #g.replay()
        #torch.cuda.synchronize()
        global graph_dict
        graph_dict[inputx.shape] = (g,inputx)
        global shape_dict
        shape_dict[inputx.shape] = 1

    def forward(self,x):
        global shape_dict
        global graph_dict
        if x.shape[0]>self.graph_bs:
            work = self.group.allreduce([x], self.ropts)
            work.wait()
        elif x.shape not in shape_dict.keys():
            #print('nccl',x.shape)
            #shape_dict[inshape] = 1
            self.warmup_and_graph(x)
            work = self.group.allreduce([x], self.ropts)
            work.wait()
        else:
            #work = self.group.allreduce([x], self.ropts)
            #work.wait()
            g,inx = graph_dict[x.shape]
            inx.copy_(x)
            g.replay()
            x = inx

        #work = self.group.allreduce([x], self.ropts)
        #work.wait()
        return x
