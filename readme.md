# NVFP4 Dequantizer

- [NVFP4 Dequantizer](#nvfp4-dequantizer)
  - [The project](#the-project)
    - [NVFP4 layout (ModelOpt)](#nvfp4-layout-modelopt)
    - [SIMD](#simd)
  - [The architecture](#the-architecture)
  - [The Multi-Core Scheme](#the-multi-core-scheme)
    - [Plate size: 256 blocks](#plate-size-256-blocks)
    - [Threshold for the workers: 16384 blocks](#threshold-for-the-workers-16384-blocks)
    - [L3 cache](#l3-cache)
  - [Final throughput](#final-throughput)
    - [Setup](#setup)
  - [Open questions](#open-questions)
  - [Fair AI Contribution](#fair-ai-contribution)

## The project

This is a streaming NVFP4 dequantizer written in Zig 0.17. I took 0.17 because it's the latest stable version, and Zig is only going to move forward, hence I'd rather adapt to the new paradigms now. It reads a .safetensors file and streams out another .safetensors file where the NVFP4 tensors are turned into F32, through a std.Io.Reader.

- Streaming: I only hold one mini-job in memory at a time, meaning the model can be as big as it wants.
- Multicore: the dequantize mini-jobs are split between several workers.

All day-to-day improvements and learnings are in `logs.md`.

### NVFP4 layout (ModelOpt)

- weight (U8, shape [rows, cols]), two E2M1 values per byte. The low nibble comes first in file order, the high nibble second. Two U4 in a U8.
- weight_scale (F8_E4M3, shape [rows, cols/8]), one scale per block of 16 values U4, which is 8 U8.
- weight_scale_2 (F32, a scalar), one F32 per tensor.
- Floating conversion, Value = e2m1 × e4m3 × scale2.

### SIMD

![Formats](<Claude%20Benchmark/formats/formats.png>)

- **E2M1:** I split each byte with `& 0x0F` for the low nibble and `>> 4` for the high one, then `@shuffle` puts them back in file order. From there I build the F32 bits directly, 16 lanes at a time. The sign goes to bit 31. For a normal value it's $E′ = e + 126$ and $M′ = m \times 2^{22}$. A subnormal (e = 0) can only be 0 or 0.5, so its bits are just `m × (126 << 23)`. Then `@select` takes one or the other depending on e.
- **E4M3:** same idea, shifts and masks straight into F32 ($E′ = e + 120$, $M′ = m \times 2^{20}$), 16 lanes at a time. For the subnormals I followed the [FP16 trick from Maratyszcza](https://github.com/Maratyszcza/FP16/blob/master/include/fp16/fp16.h). `0x7F` gives NaN. Full computation is available in the code.

## The architecture

- The Dequantizer init parses the header, makes the jobs, builds the output header and allocates the buffers for one mini-job.
- A job is a tensor. Either I copy it as it is, or it's an NVFP4 group and I dequantize it. For the NVFP4 groups the job sits on the weight, and the scales don't get a job of their own. The jobs are sorted by their offset in the input, so I read the file from start to end.
- A mini-job is one call to stream(). For a copy, it reads up to chunkSize bytes (512 KiB) with readPositionalAll. For a dequantize, it reads scale2 once at the start of the job, then chunkSize/8 scales and chunkSize bytes of weights, and writes 8 times more bytes of F32 (4 MB).
- Inside a dequantize mini-job there are plates (256 blocks, taken by one worker), groups (16 blocks, their 16 scales are decoded in one E4M3 call) and blocks (16 values).
- buildHeader rewrites the header. A copied tensor keeps its dtype and shape and only gets new offsets. A dequantized one becomes F32, its last dimension doubles and it takes 8 times more bytes. The scales are gone, \_\_metadata\_\_ stays, and the JSON is padded with spaces to a multiple of 8 like safetensors/src/tensor.rs does.
- The header is streamed first, straight to the writer, then come the jobs.

## The Multi-Core Scheme

![Scheme](<Claude%20Benchmark/scheme/scheme.png>)

I see it as a buffet, it felt a lot like playing Overcooked. The blocks of a mini-job are on the table, already cut into plates. The workers share one atomic counter, each one calls fetchAdd(1) to get the number of the next plate, decodes it and writes it at the same position in the output (times 8, since it's F32). There is no lock, and a worker that goes faster just takes more plates.

### Plate size: 256 blocks

A plate is made of groups of 16 blocks, because the E4M3 decode works on 16 lanes. The 16 scales of a group are decoded in one go, multiplied by scale2, and then each block uses its own factor on its 16 E2M1 values. Hence a plate has to be a whole number of groups, otherwise the last group of every plate would be short and padded with zeros, wasting lanes of the decode in the middle of a mini-job (the code only does that for the very last group).

Each time a worker wants a plate it calls `fetchAdd` on the counter, and that's the only place where the workers synchronize. With the smallest plate possible, one group, it's one `fetchAdd` every 16 blocks (1 KiB of output), meaning the workers spend a lot of their time waiting on each other for the counter. 256 blocks is 16 groups, so 16 KiB of F32 per plate, which I estimate at around a microsecond on one core, while the `fetchAdd` is in the tens of nanoseconds. At that point I can ignore it. 1024 blocks didn't give anything more, and bigger plates make the end of a mini-job unbalanced, since the last worker can still be on its plate while the others have nothing left to take.

### Threshold for the workers: 16384 blocks

With `minBlocksForWorkers = 0` the workers always run, even on tiny mini-jobs. At 1024 blocks one thread takes about $4.6 \mu s$ and two workers about $14.8 \mu s$, so three times slower: starting the group and waiting for it takes longer than the decoding itself. The bigger the mini-job, the more there is to decode to pay back that cost. At 8192 blocks (512 KiB of F32) everything is above one thread (14.1 GB/s), but 2 and 4 workers barely (15.3 and 15.6 GB/s), and at 16384 blocks (1 MB) they are all clearly faster, 18 to 29 GB/s against 13.9 GB/s. Below 16384 blocks I just decode on the calling thread.

### L3 cache

A dequantize mini-job of N blocks needs 8N bytes of weights, N bytes of scales and 64N bytes of F32. My Ryzen 5 7535U has 16 MB of L3 shared between its six cores. As long as the three buffers fit in there, the plates are written in cache, and since the next mini-job reuses the same buffers, they stay in it. At 131072 blocks it's 1 MB + 128 KB + 8 MB, so around 9 MB, and that's where I get the best throughput, 44 GB/s with 10 workers. At 262144 blocks the output alone is 16 MB, the whole L3, and around 18 MB with the inputs. It doesn't fit anymore, everything goes through RAM and the throughput falls to 17–21 GB/s. One thread doesn't drop at all, it stays at 10.5 GB/s from 131072 to 262144 blocks, since a single core can't use the whole memory bandwidth anyway, but with several workers writing at the same time I reach it.

![Workers](<Claude%20Benchmark/workers/workers.png>)

## Final throughput

### Setup

- Warming up all CPU threads first
- Allocating one weight buffer, one scale buffer, one output buffer
- Every block is 0.25, then 0.5 so on and so forth
- Checking the output float

$$
\text{throughput (GB/s)} = \frac{64 \times N_{\text{blocks}}}{t_{\text{call}}\ \text{(ns)}}
$$

where 64 is the number of bytes written per block (16 values of 4 bytes), and $t_{\text{call}}$ is the time of one `bufferToWorker` call.

One thread decodes 12.7 GB/s of F32. With the current mini-job of 65536 blocks, 5 workers get to 38.7 GB/s, so about 3x. The best I measured is 44 GB/s at 131072 blocks with 10 workers, which is 4.2x the 10.5 GB/s of one thread at that size. These numbers are only for the decoding, without reading the file or writing to stdout.

![Throughput](<Claude%20Benchmark/throughput/throughput.png>)

## Open questions

- If this was a real application, the question of how to serve the data would be to tackle: how to give it properly to another app to be fed properly into it. Dequantizing significantly increases the size of the data hence the receiver should be able to handle it.
- Several benchmarks are missing: the key ingredient to speed is to identify the bottleneck, understand why it's there and how to make it go. In my code, there are several moments where I don't benchmark and I don't assess the speed. It is a choice, I wanted to learn foremost instead of making it efficient.

There are two kind of main operations here: copy & dequantize. From the safetensors headers of a few public checkpoints, the split depends on the model:

| Checkpoint                        | Input size | Copy operations |
| --------------------------------- | ---------- | --------------- |
| nvidia/Qwen3-8B-FP4               | 6.4 GB     | 38.9%           |
| nvidia/Llama-3.3-70B-Instruct-FP4 | 42.7 GB    | 9.8%            |
| nvidia/DeepSeek-R1-0528-FP4       | 424 GB     | 12.7%           |

I expect multi-core to have big impact on the speed of dequantization at higher scales. Hence adding it would be a priority.

Given claude tests, starting the workers costs around 9 to 12 $\mu s$ (at 1024 blocks, 4.6 $\mu s$ on one thread against 14.8 $\mu s$ with 2 workers and 9.3 $\mu s$ with 12), meaning anything that takes less time than that for compute is utterly useless to be parallel processed.

There are two main issues that I identify:

1. The amount of data I feed to each worker is too small to pay for the price of setting them up
2. The amount of data to process overall is too little to pay for the price of setting them up.

I'm asking claude to benchmark all that because i think it's really well suited to the task.

I'm running experiments of a Ryzen 5 7535U that has a base clock of 2.9GHz on 6 cores, up to 4.55GHz on 1 core according to the data sheet. The more threads run concurrently, the more my laptop is unable to keep the frequency, leading to a slow down, cutting down the possible gains that I can have. If my 6 cores run in parallel, everything becomes slower, up to 30%, meaning that i can't get to 6x really speed up but closer to 4.2x, which is what I measured at best.

The fastest memory I can access without too much overhead is the L3 cache memory (16 MB), so 4 to 8 MB of output with the inputs on top, meaning 512KB to 1MB of chunk size. Now, what bottlenecks me is RAM.

- An output as BF16 would be more compressed as an output and would reduce the transfer size.  Perhaps there would be operations that make converting E4M3 and E2M1 to BF16 more efficient than converting to F32 and then to BF16. This is a question that I have not explored yet.
- I saw that there are two formats: ModelOpt and compressed tensors. I have selected ModelOpt due to being closer to NVIDIA, but haven't tackle how to unpack the other format.
- How do I know that the format is correct on a very large model? Here at low scale it's fine, but how could we error correct in case the weights are given wrong?
- I made my decoder turn E4M3 0x7F into NaN, as the spec says. But some other implementations dictates that 0 is tolerable leading to faster ops. To investigate. Do models have NaN anywhere?
- What about models that have different parts? How to handle tensor-1, ..., tensor-n? How to handle multiple files?
- The disk sits idle while the workers decode. A copy job is pure I/O and a dequantize job is mostly compute, so the copy jobs could run concurrently with the decoding. The gain would be on the whole stream rather than on the decode itself. The catch is the order: the output header places the tensors in job order, so a copy that runs ahead has to be held in memory until its turn in the stream.

## Fair AI Contribution

Some parts coded by Claude were credited in the files such as test arrays and speed comparisons. I also heavily requested its help to understand better the Zig 0.17 non-documentation (aka reading the source code and giving me what i'm looking for). I could not have gone that far without it for sure.
