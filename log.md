# Log file

## Project

In this file, i'll track everything that was useful to me down the line as a way to verify my sources and to cross reference what I learned with best practices.

### What is the project?

A streaming NVFP4 dequantizer written in Zig.

- **Input:** a `.safetensors` file that holds NVFP4-quantized tensors.
- **Output:** dequantized tensors, through a `std.Io.Reader`
- **SIMD** (`@Vector`): decoding of the packed FP4 values and their block scales.

The format of the output is not specified. I'll assume first that the interesting part is the dequantized values, and therefore Raw tensor bytes could meet the requirements, and fit into pipelining in another program.

Another problem: to what output format should I dequantize the values? The most common format is `f32`, but it is not the most efficient one. I could use `f16` or `bfloat16`, but they are not supported by all CPUs.

I'll start with `f32` and see how it goes.

### Definitions

Many terms that I don't know to their full extent here, so I'll take some time to elucidate them.

**NVFP4**: NVIDIA FP4, a 4-bit floating point format used in some NVIDIA GPUs for quantization.

**SIMD**: Single Instruction, Multiple Data, a parallel computing architecture that allows the same operation to be performed on multiple data points simultaneously.

**@Vector**: A Zig type that represents a vector of values, used for SIMD operations.

**Dequantization**: The process of converting quantized values back to an approximation of the original floating-point representation.

### Git maintenance

- Usage of convential commits: https://www.conventionalcommits.org/en/v1.0.0/

Commit message with description about the type of change, a scope, and a body with more details about the change. I'll follow it loosely because I keep on forgetting the exact format, but the idea is to have a clear and consistent commit history.

- No branch usage, it's a short project and I don't need to maintain multiple versions of it. So no merge requests.

## .safetensors

`.safetensors` is a file format used to store tensors in a serialized manner. It is designed to be efficient and easy to use, allowing for quick loading and saving of tensor data.

NVFP4-quantized tensors are composed of three parts back-to-back: an 8-byte length-prefixed JSON header, a JSON header and a raw block of bytes for every tensors.

- First 8 Bytes: its an unsiged 64-bit integer that represents the length of the JSON header in bytes.

> Note: This means that we can verify the integrity of the file by checking that the length of the JSON header matches the value stored in the first 8 bytes.
> This can be useful to detect corrupted files or transmission errors.

- JSON Header: a JSON object that contains metadata about the tensors, such as their shapes, data types, and quantization parameters.

Ex:

```json
{ 
"__metadata__": { 
    "format": "pt" 
  },
  "model.layers.0.mlp.up_proj.weight": { 
    "dtype": "U8",  
    "shape": [14336, 2048], 
    "data_offsets": [0, 29360128] 
  },
  "model.layers.0.mlp.up_proj.weight_scale":   { 
    "dtype": "F8_E4M3", 
    "shape": [14336, 256],  
    "data_offsets": [29360128, 33030144] },
  "model.layers.0.mlp.up_proj.weight_scale_2": { 
    "dtype": "F32",   
    "shape": [],      
    "data_offsets": [33030144, 33030148] 
  }
}
```

We have the `__metadata__` key that contains the format of the file. `pt` being pytorch.

Then we have the tensors themselves, each tensor has a name (the key), and a value that is an object that contains the `dtype`, `shape`, and `data_offsets`.

> Note: The shape can be empty, which means that the tensor is a scalar. The `data_offsets` are the offsets in the raw block of bytes where the tensor data is stored.
> Enum of types? to verify that the `dtype` is valid and supported. Is there any switch commands on Zig?
> type consistency? end - begin == product(shape) × sizeof(dtype)
> bounds? 8 + N + end ≤ file size
> No holes? tensors must cover th ewhole byte block, hence sorting the tensors by their `data_offsets` and checking that the `end` of one tensor is the `begin` of the next one?

Warning: Large models are split into several `.safetensors` files, and the tensors are split across these files. Maybe this is where multi-threading could be useful, to read the tensors in parallel from multiple files to avoid I/O bottlenecks.

### Where to get an example `.safetensors` file?

- https://huggingface.co/nvidia/Nemotron-3-Embed-1B-NVFP4: Nemotron-3-Embed-1B-NVFP4 is the quantized version of the Nemotron-3-Embed-1B-BF16 model by NVIDIA Model Optimizer.

## Zig

### Versionning

First question that I'm going to ask: What version of Zig should I use?

Due to Zig being a language that is still in development, the documentation changed a lot, hence it's gonna be a mess to find the right information.

I'll work on `0.17.0', which is the latest stable version at the time of writing, because Zig is going to be only moving forward and hence I should adapt to the new paradigms.

### Documentation

- [https://ziglang.org/documentation/0.17.0/](https://ziglang.org/documentation/0.17.0/): The official Zig documentation for my version (`master` documents the dev version)
- [https://ziglang.org/download/0.16.0/release-notes.html](https://ziglang.org/download/0.16.0/release-notes.html): The big `std.Io` rework (files, threads, sync primitives); most tutorials online predate it
- [https://ziglang.org/download/0.17.0/release-notes.html](https://ziglang.org/download/0.17.0/release-notes.html): Changes since 0.16 (`@bitCast` on vectors, build system renames)
- [gist.github.com/ityonemo/769532c2017ed9143f3571e5ac104e50](https://gist.github.com/ityonemo/769532c2017ed9143f3571e5ac104e50): a half hour to learn Zig, the basics of the language.

**Old projects from the team at ZML**

- [https://github.com/erwanvivien/nvfp4-dequant](https://github.com/erwanvivien/nvfp4-dequant)
- [https://github.com/pierrelgol/nvfp4-dequantizer](https://github.com/pierrelgol/nvfp4-dequantizer)
- [https://github.com/tgautie/zig_mxfp4_dequantizer](https://github.com/tgautie/zig_mxfp4_dequantizer)

### Zig cheat sheet

Compiling works in Zig through a `build.zig` file, which is a Zig script that describes how to build the project. The `zig build` command will run this script and produce the output in the `zig-out/` directory.

**Commands**

| Command             | What it does                                                        |
| ------------------- | ------------------------------------------------------------------- |
| `zig version`       | Show the installed version                                          |
| `zig init`          | Create `build.zig`, `build.zig.zon`, `src/main.zig`, `src/root.zig` |
| `zig build`         | Build the project into `zig-out/`                                   |
| `zig build run`     | Build, then run the executable                                      |
| `zig build test`    | Run the tests declared in `build.zig`                               |
| `zig run file.zig`  | Compile and run one file, no `build.zig` needed                     |
| `zig test file.zig` | Run the `test` blocks of one file                                   |
| `zig fmt .`         | Format all the code                                                 |

**Build flags**

| Flag                      | What it does                                      |
| ------------------------- | ------------------------------------------------- |
| `-Doptimize=Debug`        | Default: safety checks, slow                      |
| `-Doptimize=ReleaseSafe`  | Optimized, keeps safety checks                    |
| `-Doptimize=ReleaseFast`  | Fastest, no safety checks (use it for benchmarks) |
| `-Doptimize=ReleaseSmall` | Smallest binary                                   |

### Tests

Tests in Zig are written in `test` blocks, which can be placed anywhere in the code. They are run with `zig build test` first, then `zig test file.zig`. Since 0.16, tests that need I/O use `std.testing.io`.

- https://ziglang.org/documentation/0.17.0/#Zig-Test: The official testing chapter for my version.

### Multi-core

Since 0.16, concurrency goes through the `std.Io` interface: `io.async(func, .{args})` returns a `Future` that I `await` (or `cancel`), `io.concurrent(...)` is for work that must truly run in parallel (it can fail with `error.ConcurrencyUnavailable`), and `std.Io.Group` handles many tasks with one lifetime. `std.Io.Threaded` is the thread-based implementation that `main` gets by default. `std.Thread.Pool` was removed, and the sync primitives moved to `std.Io` (`std.Io.Mutex`, `Condition`, `RwLock`, `Semaphore`, `ResetEvent`; `WaitGroup` became `std.Io.Group`). Atomics are still `std.atomic.Value`.

The best practice is to need as few of these as possible: give each task its own contiguous range of the output and its own allocator, so no locks are needed, and pad per-task data to a cache line to avoid false sharing.

> `await` returns the task's result, errors included. Raw `std.Thread.spawn` + `join()` does not pass errors back.

For this project, threads come last: threading should be added only if it is a bottleneck. Hence, first I will measure speed and do my best to see how that works.

### JSON

Zig can handle JSON through `std.json`, which is a streaming parser that can parse JSON from a `std.Io.Reader` and produce a `std.json.Value` that can be traversed.

### Comptime, Switches

To select what to do based on the `dtype` of the tensor, I will use a `switch` statement.

There is a `comptime` keyword that allows to execute code at compile time, which is useful for generating code based on the `dtype` of the tensor.

## Journal

### 18/10/2026

- Documented the project and its goals, as well as the tools and resources that I will use to implement it.
- Read the `safetensors` format specification and understood how to read the header and the tensor metadata.

### To Do LIST

- Download a small NVFP4 checkpoint into `data/` and read its header by hand
- [gist.github.com/ityonemo/769532c2017ed9143f3571e5ac104e50](https://gist.github.com/ityonemo/769532c2017ed9143f3571e5ac104e50): Read a half hour to learn Zig

## Open Questions

- To what format should I dequantize the values? `f32` is the most common format, but it is not the most efficient one. I could use `f16` or `bfloat16`, but they are not supported by all CPUs/GPUs.
  I'll start with `f32` and see how it goes, and defend it if I need to.
- To what output format should I dequantize the stream? Raw tensor bytes could meet the requirements, and fit into pipelining in another program. But I need to verify that this is the case.

Or another `safetensors` file, but that would require a lot of work to implement, and I don't know if it is worth it. I'll start with raw tensor bytes and see how it goes.

- What to do with non-NVFP4 tensors if any?
