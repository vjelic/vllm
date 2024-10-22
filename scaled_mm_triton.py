import torch
import triton
import triton.language as tl


@triton.jit
def scaled_mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr,
                     BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                     SPLIT_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)

    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    # num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = tl.float32

    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    # accumulator = tl.arange(0, BLOCK_SIZE_N)
    # accumulator = tl.broadcast_to(accumulator[None, :],
    # (BLOCK_SIZE_M, BLOCK_SIZE_N))
    # accumulator = accumulator & 0x0
    # accumulator = accumulator.to(accumulator_dtype)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                           dtype=accumulator_dtype)

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    offsets_a = K * offsets_am[:, None] + offsets_k[None, :]
    offsets_b = N * offsets_k[:, None] + offsets_bn[None, :]

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    # NOTE: Use this in TRITON_INTERPRET=1 mode instead of tl.cdiv
    # block_offset = BLOCK_SIZE_K * SPLIT_K
    # for k in range(0, (K + block_offset - 1) // (block_offset)):
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a).to(tl.float32)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b).to(tl.float32)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K * SPLIT_K
        a_ptrs += BLOCK_SIZE_K * SPLIT_K
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * N

    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_cm = offs_cm.to(tl.int64)
    offs_cn = offs_cn.to(tl.int64)
    # c_ptrs = c_ptr + pid_z * N * M + N * offs_cm[:, None] + offs_cn[None, :]
    # c_ptrs = c_ptr + pid_z * N * M + N * offs_cm[:, None] + offs_cn[None, :]
    c_ptrs = c_ptr + pid_z * N * M + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    c = c.to(tl.float32)
    tl.store(c_ptrs, c, mask=c_mask)


# input   - [M, K]
# weight - [K, N]
# split_k_iters - parallelism along K-dimension, int, power of 2.
def scaled_mm_triton(input: torch.Tensor,
                     weight: torch.Tensor,
                     block_size_m: int = 32,
                     block_size_n: int = 32,
                     block_size_k: int = 32) -> torch.Tensor:
    M, K = input.shape
    N = weight.shape[1]

    assert N > 0 and K > 0 and M > 0
    assert weight.shape[0] == K

    split_k_iters = 1

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(
            N, META['BLOCK_SIZE_N']),
        split_k_iters,
    )

    result = torch.zeros((split_k_iters, M, N),
                         dtype=torch.float32,
                         device=input.device)

    # print(f"scaled_mm_kernel:(before call)result.dtype = {result.dtype}")
    # A = input, B = weight, C = result
    # A = M x K, B = K x N, C = M x N
    scaled_mm_kernel[grid](input,
                           weight,
                           result,
                           M,
                           N,
                           K,
                           BLOCK_SIZE_M=block_size_m,
                           BLOCK_SIZE_N=block_size_n,
                           BLOCK_SIZE_K=block_size_k,
                           SPLIT_K=split_k_iters)

    # print(f"scaled_mm_kernel:result.dtype = {result.dtype}")
    result = result.sum(0, dtype=torch.float32)

    # print(f"=================result.shape = {result.shape}")
    return result


# a.shape = torch.Size([1, 17920]), b.shape = torch.Size([17920, 5120]),scale_a.shape = torch.Size([1, 1]), scale_b.shape = torch.Size([5120, 1]),bias.shape = None
# a.shape = torch.Size([1, 5120]), b.shape = torch.Size([5120, 35840]),scale_a.shape = torch.Size([1, 1]), scale_b.shape = torch.Size([35840, 1]),bias.shape = None
# a.shape = torch.Size([1, 5120]), b.shape = torch.Size([5120, 5120]),scale_a.shape = torch.Size([1, 1]), scale_b.shape = torch.Size([5120, 1]),bias.shape = None
# a.shape = torch.Size([1, 5120]), b.shape = torch.Size([5120, 7680]),scale_a.shape = torch.Size([1, 1]), scale_b.shape = torch.Size([7680, 1]),bias.shape = None
# a.shape = torch.Size([131072, 17920]), b.shape = torch.Size([17920, 5120]),scale_a.shape = torch.Size([131072, 1]), scale_b.shape = torch.Size([5120, 1]),bias.shape = None
# a.shape = torch.Size([131072, 5120]), b.shape = torch.Size([5120, 35840]),scale_a.shape = torch.Size([131072, 1]), scale_b.shape = torch.Size([35840, 1]),bias.shape = None
# a.shape = torch.Size([131072, 5120]), b.shape = torch.Size([5120, 5120]),scale_a.shape = torch.Size([131072, 1]), scale_b.shape = torch.Size([5120, 1]),bias.shape = None
# a.shape = torch.Size([131072, 5120]), b.shape = torch.Size([5120, 7680]),scale_a.shape = torch.Size([131072, 1]), scale_b.shape = torch.Size([7680, 1]),bias.shape = None
# a.shape = torch.Size([15, 17920]), b.shape = torch.Size([17920, 5120]),scale_a.shape = torch.Size([15, 1]), scale_b.shape = torch.Size([5120, 1]),bias.shape = None
# a.shape = torch.Size([15, 5120]), b.shape = torch.Size([5120, 35840]),scale_a.shape = torch.Size([15, 1]), scale_b.shape = torch.Size([35840, 1]),bias.shape = None
# a.shape = torch.Size([15, 5120]), b.shape = torch.Size([5120, 5120]),scale_a.shape = torch.Size([15, 1]), scale_b.shape = torch.Size([5120, 1]),bias.shape = None

# a.shape = torch.Size([15, 5120]), b.shape = torch.Size([5120, 7680]),
# scale_a.shape = torch.Size([15, 1]), 
# scale_b.shape = torch.Size([7680, 1]),bias.shape = None

# a.shape = torch.Size([1, 17920]), a.dtype = torch.int8,
# b.shape = torch.Size([17920, 5120]), b.dtye = torch.int8,
# scale_a.shape = torch.Size([1, 1]), scale_a.dtype = torch.float32,
# scale_b.shape = torch.Size([5120, 1]), scale_b.dtype = torch.float32,
# bias.shape = None,bias.dtype = None
def main():
    test_cases = [
        # M        K     N
        (1, 17920, 5120),
        (1, 5120, 35840),
        (1, 5120, 5120),
        (1, 5120, 7680),
        (131072, 17920, 5120),
        (131072, 5120, 35840),
        (131072, 5120, 5120),
        (131072, 5120, 7680),
        (15, 17920, 5120),
        (15, 5120, 35840),
        (15, 5120, 5120),
        (15, 5120, 7680),
    ]

    import time

    for test_case in test_cases:
        M, K, N = test_case
        a = torch.randint(0, 127, (M, K), dtype=torch.int8, device='cuda')
        b = torch.randint(0, 127, (K, N), dtype=torch.int8, device='cuda')
        print("=" * 5 + f" Testing: scaled_mm_triton M={M}, K={K}, N={N}" +
              "=" * 5)
        start = time.time()
        c_check = scaled_mm_triton(a, b)
        end = time.time()

        print(f"c_check time: {end - start}")
        print(f"c_check.dtype = {c_check.dtype}")

        a_cpu = a.cpu()
        b_cpu = b.cpu()
        start = time.time()
        # c_actual = torch.mm(a_cpu.to(torch.float), b_cpu.to(torch.float))
        c_actual = torch._int_mm(a_cpu, b_cpu)
        end = time.time()

        # c_actual = c_actual.to(torch.int)
        print(f"c_actual time: {end - start}")
        print(f"c_actual.dtype = {c_actual.dtype}")

        print(f"c_check = {c_check}")
        print(f"c_actual = {c_actual}")

        print(f"equal: {torch.equal(c_check.cpu(), c_actual)}")


if __name__ == "__main__":
    main()
