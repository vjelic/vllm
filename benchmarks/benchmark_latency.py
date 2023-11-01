"""Benchmark the latency of processing a single batch of requests."""
import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
import vllm
import ray

print('>>>',vllm.__file__)


def main(args: argparse.Namespace):
    print(args)

    # Process all the requests in a single batch if possible.
    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.batch_size * args.input_len,
        trust_remote_code=args.trust_remote_code,
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0 if args.use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    dummy_prompt_token_ids = [[33] * args.input_len] * args.batch_size
    dummy_prompts = []
    dummy_prompts.append('DeepSpeed is a machine learning library that deep learning practitioners should use for what purpose')
    dummy_prompts.append ("As a highly detailed prompt generator for a still image generative AI, your task is to create 9 intricate prompts for each of the provided concept that vividly describes a photo.   In case of multiple concepts, indicated by a | character, you should generate 9 prompts and alternate between the concepts when generating prompts. For instance, if I give you dog or cat as a concept, you should create the first prompt for a dog, the second for a cat, the third for a dog, and so on.   Each prompt should consist of 50 to 70 words and follow a structure that Initiate each prompt with something and incorporate an appropriate phrase such as Photography of, Wartime Photography of, Food Photography of, or similar that best portrays the concept. Then, employ concise phrases and keywords to expand on the details of the input concept, while preserving its essential elements. Include relevant props or objects in the scene to add depth and context. Describe these objects using short phrases and keywords. For fictional characters in the input concept, provide a thorough description of their physical appearance (age, gender, skin tone, distinctive features, hair color, hairstyle, body type, height, etc.), emotional state, actions/behavior, and anything else noteworthy. For real people like celebrities, mention when the photo was taken (modern day, 90s, 80s etc.), their emotional state, and actions/behavior.   Example: Food Photography of a mouthwatering chocolate cake. Another example describes the environment/background using short phrases and keywords. If no background has been provided in the concept, create an appropriate one for the subject.   Example: displayed on an antique wooden table, If applicable, describe the relationships, interactions, or contrasts between multiple subjects or elements within the photograph.   Example: showcasing the juxtaposition of old and new architectural styles, incorporate realistic descriptions of the photo concept: framing (close-up, wide shot, etc.), angle (low angle, high angle, etc.), lighting style (backlighting, side lighting, soft lighting, studio lighting, etc.), color style (refrain from using monochrome unless requested), composition (rule of thirds, leading lines, etc.). Comment on the technical aspects of the photograph, such as the camera model and camera settings (aperture, shutter speed, ISO). Describe the post-processing techniques of the photo or effects, such as filters, vignettes, or color grading, that enhance the visual impact of the image.   Example: 2 close")
    #print('>>>Num Prompts',len(dummy_prompts))
    def run_to_completion(profile: bool = False):
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.time()

        rsp = llm.generate(
                     prompt_token_ids=dummy_prompt_token_ids,    # prompts=dummy_prompts, #prompt_token_ids=dummy_prompt_token_ids,
                     sampling_params=sampling_params,
                     use_tqdm=False)

        end_time = time.time()
        latency = end_time - start_time
        print('>>>Rsp',rsp[0].outputs)
        #print('>>>Rsp',rsp[1].outputs)
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return latency

    print("Warming up...")
    run_to_completion(profile=False)

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile=False))
    print(latencies)
    print(f'Avg latency: {np.mean(latencies)} seconds')
    del llm
    ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
                    'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n', type=int, default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters', type=int, default=3,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code', action='store_true',
                        help='trust remote code from huggingface')
    args = parser.parse_args()
    main(args)
    #time.sleep(30) #add sleep for profiling
