# NVFP4 Dequantizer




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

- An output as BF16 would be more compressed as an output and would reduce the transfer size.  Perhaps there would be operations that make converting E4M3 and E2M1 to BF16 more efficient than converting to F32 and then to BF16. This is a question that I have not explored yet.

- I saw that there are two formats: ModelOpt and compressed tensors. I have selected ModelOpt due to being closer to NVIDIA, but haven't tackle how to unpack the other format.

- How do I know that the format is correct on a very large model? Here at low scale it's fine, but how could we error correct in case the weights are given wrong?

- I made my decoder turn E4M3 0x7F into NaN, as the spec says. But some other implementations dictates that 0 is tolerable leading to faster ops. To investigate. Do models have NaN anywhere?

- What about models that have different parts? How to handle tensor-1, ..., tensor-n? How to handle multiple files?

- I didn't reconstruct the header of the safetensors file so far. I decided to drop it for the sake of time. (the project didn't mention the format).

## Fair AI Contribution

Some parts coded by Claude were credited in the files such as test arrays and speed comparisons. I also heavily requested its help to understand better the Zig 0.17 non-documentation (aka reading the source code and giving me what i'm looking for). I could not have gone that far without it for sure.
