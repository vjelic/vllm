vLLM Benchmarking Recommendations
=====================================

vLLM offers a suite of benchmarking scripts that cover multiple scenarios, which can be divided into three main categories:

1.  Offline latency measurement for fixed input and output sizes.
2.  Offline throughput measurement on a sample of a dataset, simulating server behavior.
3.  Server mode benchmark for TTFT, TPOT, and other server metrics.

### Latency Benchmark

**benchmark_latency.py**

This script is used to measure the latency of vLLM for a fixed input and output size. It takes the following parameters:

*   `--batch-size`: The batch size to use for the benchmark.
*   `--input-len`: The fixed input length to use for the benchmark.
*   `--output-len`: The fixed output length to use for the benchmark.
*   `--num-iters-warmup`: The number of warm-up iterations to run before measuring latency.
*   `--num-iters`: The number of iterations to run for measuring latency.

The benchmark returns the time required to compute the result sequence based on the input parameters.

**Important Note:** People often mistakenly use `benchmark_throughput.py` for measuring fixed datasets. To correctly get the throughput numbers for a fixed size set, use `benchmark_latency.py` and process the result as follows:

*   `input tokens/s = (batch-size * input-len) / latency`
*   `output tokens/s = (batch-size * output-len) / latency`
*   `total tokens/s = (batch-size * (input-len + output-len)) / latency`

**Common Mistake:** Do not use `--output-len` equal to 1 to try and simulate TTFT. This will not provide accurate results.

### Throughput Benchmark

**benchmark_throughput.py**

This script is used to measure the throughput of vLLM on a sample of a dataset, simulating server behavior. It takes the following parameters:

*   `--dataset-name`: The name of the dataset to use (e.g., `random`, `sharegpt`, etc.).
*   `--dataset-path`: The path to the dataset.
*   `--num-prompts`: The number of prompts to use from the dataset.
*   `--output-len`: The output length to use for the benchmark (overrides the output length from the dataset).
*   `--async-engine`: Use the async vLLM engine to better simulate server mode behavior.

**Important Note:** Do not try to use this benchmark for measuring fixed size input/output datasets. Use `benchmark_latency.py` for this purpose.

### Serving Benchmark

**benchmark_serving.py**

This script is used to measure the actual server mode performance and metrics relevant to online serving, such as Time To First Token (TTFT). It is run together with the vLLM server.

**Important Note:** This is the only benchmark that allows reliably and correctly measuring the TTFT metric.

### Useful Links

*   [vLLM Documentation: Contributing/Benchmarks](https://docs.vllm.ai/en/latest/contributing/benchmarks.html)
*   [vLLM GitHub: Performance Benchmarks Descriptions](https://github.com/vllm-project/vllm/blob/main/.buildkite/nightly-benchmarks/performance-benchmarks-descriptions.md)

By following these guidelines and using the correct benchmarks for your use case, you can ensure accurate and reliable results when measuring the performance of vLLM.