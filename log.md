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

### The crux of this project: FP4 dequantization

NVFP4 combines three number formats: 4-bit values (E2M1), an 8-bit scale per block (E4M3), and a 32-bit scale per tensor (F32).

[docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.14/user-guide/features/low_precision_training/nvfp4/nvfp4.html](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.14/user-guide/features/low_precision_training/nvfp4/nvfp4.html): docs from nvidia

#### FP4 E2M1 (the values)

- 4 bits: `s e e m` (1 sign, 2 exponent, 1 mantissa), exponent bias 1.
- Normal (`e != 0`): `(-1)^s * 2^(e - 1) * (1 + m/2)`
- Subnormal (`e == 0`): `(-1)^s * 0.5 * m` (so `0` or `0.5`)
- Only 16 possible codes, 8 magnitudes: `0, 0.5, 1, 1.5, 2, 3, 4, 6`. Max = 6.
- Packed two per byte (stored as `U8` in safetensors).

They're usually stored in uint8, containing 4 bits of lower nibble and 4 bits of upper nibble. If a byte has the hexadecimal value 0x71 (binary 0111 0001):

- Lower nibble (0x1 / 0001): Represents the first E2M1 element (value: 0.5).(subnormal)
- Upper nibble (0x7 / 0111): Represents the second E2M1 element (value: 6.0). (normal)

#### FP8 E4M3 (the block scales)

- 8 bits: 1 sign, 4 exponent, 3 mantissa, exponent bias 7.
- Normal (`e != 0`): `(-1)^s * 2^(e - 7) * (1 + m/8)`
- Subnormal (`e == 0`): `(-1)^s * 2^(-6) * (m/8)`
- No infinity. NaN = `S.1111.111` only, so `e == 15` is still a valid normal exponent for the other mantissas.
- Max = 448 (`0.1111.110`), smallest subnormal = 2^-9.
- There seems to be confusion about E4M3: is it unsigned or signed? I can only see E4M3 in the docs as a signed implementation. I'll keep it signed for now. (https://docs.nvidia.com/cuda/pdf/ptx_isa_9.2.pdf)

#### NVFP4 (NVIDIA)

- Blocks of **16** E2M1 values sharing one **E4M3** scale, plus one **F32** scale per tensor.
- `value = e2m1 * e4m3_block_scale * f32_tensor_scale`
- In a ModelOpt-style checkpoint:
  - `X.weight`: `U8`, shape `[rows, cols/2]` (packed nibbles)
  - `X.weight_scale`: `F8_E4M3`, shape `[rows, cols/16]`
  - `X.weight_scale_2`: `F32`, scalar
- Compared with MXFP4: smaller blocks and a scale with real mantissa bits (E4M3 vs a pure power of two) give finer scaling, so lower error. The F32 tensor scale makes up for E4M3's limited range (max 448).

## The maths behind the operations of bit shift

Huge shoutout to @Théo Ballet ([www.linkedin.com/in/th%C3%A9o-b-6a1714130](https://www.linkedin.com/in/th%C3%A9o-b-6a1714130/)), my amazing hardware guy for walking me through the operations.

There's this cool website about many format too ([sw23.github.io/fp-conv/formats/fp32.html](https://sw23.github.io/fp-conv/formats/fp32.html))!

Each field is a number multiplied by the weight of its lowest bit:

| format | layout                                   | as a sum              |
| ------ | ---------------------------------------- | --------------------- |
| E2M1   | `s ee m`                                 | s·2^3 + e·2^1 + m     |
| E4M3   | `s eeee mmm`                             | s·2^7 + e·2^3 + m     |
| f32    | `s` + 8 exponent bits + 23 mantissa bits | s·2^31 + E′·2^23 + M′ |

They all have a bias, respectively 1, 7 and 127. To convert one into another, we have to set equalities behind the different concerned bits.

- Sign: s' = s, so we have to move it to the top of the bit (<<31)
- Exponent: 2 ^(E' - 127) = 2^(e-bias) so E' = e + (127 - bias)
- Mantissa: M' / 2^23 = m/2^M so M' = m * 2^(23 - M)

Hence, we just have to fill each format

|      | sign shift                               | E′      | M′                     |
| ---- | ---------------------------------------- | ------- | ---------------------- |
| E2M1 | 3 → 31: `<< 28` (or `>> 3` then `<< 31`) | e + 126 | m · 2^22, so `m << 22` |
| E4M3 | 7 → 31: `<< 24`                          | e + 120 | m · 2^20, so `m << 20` |

Because exponents are above mantissa, we can also move both together as a group and combine operations.

Most of the issue comes from subnormals that break the formula. For E2M1 it's simple because m can only take 2 values yielding 0 or 0.5: `m * (126 << 23)`

But for E4M3, it's messier. [github.com/Maratyszcza/FP16/blob/master/include/fp16/fp16.h](https://github.com/Maratyszcza/FP16/blob/master/include/fp16/fp16.h) provides an implementation for float16. Here, it's 8 values. They propose to build and substract by

1. Pretending e=1 and compute 2^-6 (1+ m/8)
2. Subtract $2^{-6}$, yielding $2^{-6} \cdot \frac{m}{8}$ with no rounding error, by **Sterbenz's lemma**: if $\frac{y}{2} \le x \le 2y$, the float result of $x - y$ is exact.

> Here $x = 2^{-6}\left(1 + \frac{m}{8}\right)$ and $y = 2^{-6}$, so $y \le x \le 1.875 \cdot y$ and the condition holds.

### Timing tables

I measured the timing between implementations by asking Claude to make a little benchmark comparator (i'm not too familiar with the comptime and pointers over functions for now, so i'm delegating because it's a faraminous amount of data at the same time to process. It was computed on Ryzen 7 5800X.

#### E2M1

| variant                           | ns/code |
| --------------------------------- | ------- |
| `e2m1ToF32SimdSlow` (`@exp2`)     | 2.13    |
| `e2m1ToF32SimdLut` (`inline for`) | 0.34    |
| `e2m1ToF32Simd` (bits)            | 0.16    |

Exponential is always super slow to compute unfortunately.

#### E4M3

| variant                         | ns/code  |
| ------------------------------- | -------- |
| scalar `e4m3ToF32` (`math.pow`) | 16.1     |
| LUT                             | 0.28     |
| **`e4m3ToF32bits`**             | **0.13** |

I didn't code the LUT, but i'll take the word of claude for it: i coded the simd lut for e2m1, it must be pretty similar.

Open questions:

- Once the compute is approximately at the same speed for both methods we still have our formula **value = E2M1 code × E4M3 block scale × f32 tensor scale.** As we are waiting for both these instructions to be computed to multiply them, the objective is to have the both of them as low as possible at the same time. But it would be good to check if the bottleneck is in the compute or in the ram before trying to make it faster. What is bottlenecking?

When coding the full pipeline for dequantizing, I realized that `@Vector(16, u8)` for the E4M3 is maybe not the smartest because while NVFP4 blocks are decoded 16 by 16, only one E4M3 is needed for this. So the best I could do is actually to pick the width at `comptime` depending on the size of my buffer.

My current idea of a scheme would be:

- Split blocks into K ranges of 16 blocks
- Each worker takes a range of 16 blocks ($8\times16$ for the weights, 16 scales 1, 1 scale 2, and outs $64\times16$
- Wait for all worker
- Go to the next minijob

#### The nibble conundrum

Take a full u8 array of weights gathered from your buffer. each u8 entry is a u4, with low and high nibble back to back. When splitting, we retrieve a vector on only the low nibbles and only the high. Then we must make an entire vector that has first a low nibble then a high nibble so on and so forth.

This is where the simd function @shuffle comes from, allowing us to actually permute the entries of a vector according to a mask.

[ziglang.org/documentation/0.17.0/#shuffle](https://ziglang.org/documentation/0.17.0/#shuffle)

## Zig

### Versionning

First question that I'm going to ask: What version of Zig should I use?

Due to Zig being a language that is still in development, the documentation changed a lot, hence it's gonna be a mess to find the right information.

I'll work on `0.17.0', which is the latest stable version at the time of writing, because Zig is going to be only moving forward and hence I should adapt to the new paradigms.

### Documentation

- [https://ziglang.org/documentation/0.17.0/](https://ziglang.org/documentation/0.17.0/): The official Zig documentation for my version (`master` documents the dev version)
- [https://ziglang.org/download/0.16.0/release-notes.html](https://ziglang.org/download/0.16.0/release-notes.html): The big `std.Io` rework (files, threads, sync primitives)
- [https://ziglang.org/download/0.17.0/release-notes.html](https://ziglang.org/download/0.17.0/release-notes.html): Changes since 0.16 (`@bitCast` on vectors, build system renames)
- [gist.github.com/ityonemo/769532c2017ed9143f3571e5ac104e50](https://gist.github.com/ityonemo/769532c2017ed9143f3571e5ac104e50): a half hour to learn Zig, the basics of the language.
- [mojoauth.com/dev-guides/serialize-and-deserialize-json-with-zig](https://mojoauth.com/dev-guides/serialize-and-deserialize-json-with-zig): deserialize json with zig
- [Ziggit: How to parse arbitrary JSON data with std.json?](https://ziggit.dev/t/how-to-parse-arbitrary-json-data-with-std-json/13437): parse an arbitrary json with zig
- [Hugging Face: safetensors metadata parsing](https://huggingface.co/docs/safetensors/metadata_parsing): safetensor parsing

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

For functions returns ! means an error can be returned, ? means null can be returned. !? means maybe both, and you can state the typing with it !?Nvfp4.

### JSON

Zig can handle JSON through `std.json`, which is a streaming parser that can parse JSON from a `std.Io.Reader` and produce a `std.json.Value` that can be traversed.

### Structures

Structures can be made much like in c and c++.

```zig
pub const Job = struct {
    name: []const u8,
    info: TensorInfo,
    action: Action,
};
```

They can also hold pointers on functions that kind of look like methods from OOP classes, allowing to deinit memory if needed.

### Reading of a file

Reading the file sequentially is not a good way to do it, because the file can be large and the tensors can be scattered across the file. If I was to store every position of the tensors in memory, then act of the arrays one by one, it wouldn't be efficient.

Chunking seems to be the good intuition: NVFP4 takes blocks of 16 values, and the scales are stored in blocks of 16 as well.

How do I retrieve the scales for each block first, then the values for each block?

I could put one reading pointer at the sector of the scales, and one reading pointer at the sector of the values, and read them in parallel. The sector of the values needs to have chunking, because more values could be complicated to handle.

Scheme:

1. Retrieve the header informations.
2. Filter by NVFP4
3. Retrieve the list of NVFP4 to process
4. Sort the list by block
5. Read sequentially the jobs (we're producing a stream)
6. Split the jobs into chunks
7. Dispatch the chunks to be multiprocessed?

### Streaming

Ok let's break down what needs to be done here in terms of streaming. The job array is a plan: it depicts what it needs to do and where to do it, and nothing is read. Only when I call the stream will I process the jobs.  So i need only positions to be stored.

Phases:

1. Build the jobs (dispatch) :
   1. If its a nvfp4 triplet, i mark it dequantize
   2. if its a scale, skip
   3. if its anything else, copy and hold the position of bytes
2. Remove the skip jobs
3. Sort the jobs by position
4. Build the output header of the job
5. Stream
   1. Give the header + metadata
   2. Perform the jobs sequentially
      1. Copy: just hand out the bytes (can use mini-jobs later)
      2. Dequantize (can use mini-jobs later):
         1. Read the scales
         2. Read the weights
         3. Dequantize them into f32 and handle them out

Turns out that one of the worst things in a programmmer's life is to find out the lack of documentation on a library due to a release just being done.

The full documentation of the `std.Io.Reader` interface is here: https://github.com/ziglang/zig/blob/master/lib/std/Io/Reader.zig

From what I can collect: there's four fields in this reader:

```zig
vtable: *const VTable,
buffer: []u8,
/// Number of bytes which have been consumed from `buffer`.
seek: usize,
/// In `buffer` before this are buffered bytes, after this is `undefined`.
end: usize,
```

Buffer, seek and end are self explanatory, but vtable is not. Vtable is a struct that holds function pointers implementing the reader interface. It has one thing that is what interests us: "stream".

```zig
stream: *const fn (r: *Reader, w: *Writer, limit: Limit) StreamError!usize,
```

So it takes in a reader, a writer and a limit.

> /// Implementations are encouraged to utilize mandatory minimum buffer
> /// sizes combined with short reads (returning a value less than `limit`)
> /// in order to minimize complexity.

Definitely, our process job must take place within stream.

Line 260, ``pub fn streamRemaining(r: *Reader, w: *Writer) StreamRemainingError!usize `` is used to set the stream. It just has an offset and and calls the stream until it's done.

Now, how does the reader provision its buffer? File seeking can be done using readPositionalAll: https://www.ziglang.in/learn/files/random-access/, https://www.mtsoukalos.eu/2026/04/file-seeking-in-zig/.

### Multi-core

Claude resume:

Since 0.16, concurrency goes through the `std.Io` interface: `io.async(func, .{args})` returns a `Future` that I `await` (or `cancel`), `io.concurrent(...)` is for work that must truly run in parallel (it can fail with `error.ConcurrencyUnavailable`), and `std.Io.Group` handles many tasks with one lifetime. `std.Io.Threaded` is the thread-based implementation that `main` gets by default. `std.Thread.Pool` was removed, and the sync primitives moved to `std.Io` (`std.Io.Mutex`, `Condition`, `RwLock`, `Semaphore`, `ResetEvent`; `WaitGroup` became `std.Io.Group`). Atomics are still `std.atomic.Value`.

> `await` returns the task's result, errors included. Raw `std.Thread.spawn` + `join()` does not pass errors back.

If I want to make it multi-core, I must plan ahead and figure out the most efficient way to do it, because a bad plan leads to a nightmare. I'm kinda up for the task of making a batch-like yielder. I don't think I'll make it much more efficient because the time to get it better is very high.

First, we need to know what to do with the types that are of no interest to the dequantizer. Once we have our structure, we can create an array of job with some kind of "action" and depending on the action, we know what to yield to the reader

What I'm currently thinking of is that:

- We have an array of jobs
- A job can be composed of minijobs: a minijob holds scale_1, scale_2, the position of the weights
- It raises the question of how to keep the ordering consistent, what if a job that comes later finishes first? Naive implementation: we have batches of roughly the same size, the compute time should stay consistent, it's okay to wait for the first time and pay the eventual overhead of the end of the batch.
- How do we face the reading contention? Maybe we initialize a reader with a big buffer and then dispatch to the cpu
- To read the position of the weights, there's the function `seekTo` in the reader.

What we can have is then:

- An array of jobs
- A for jobs in array,
- One job,
- A while the job isn't finished,
- A big reader taking a big slot of the memory (probably a function of the number of cpu/available threads like 8xN),
- Dispatch the reads to workers by creating a minijob that has scale_1, scale_2 and the position of the weights it has to compute,
- Wait for each thread sequentially to come back
- Increase the position of the big reader
- Finish while
- Finish for
- Close the queue

Raw f32 of the NVFP4 tensors, in file order? What happens to the BF16 tensors, which are half of the real file?

### Comptime, Switches

To select what to do based on the `dtype` of the tensor, I will use a `switch` statement.

There is a `comptime` keyword that allows to execute code at compile time, which is useful for generating code based on the `dtype` of the tensor.

### Timing

In Zig v0.17 , timing is done through `std.Io.Timestamp`, which can be used to measure the time taken by a block of code. It changed from `std.time` in v0.16.

```zig
const start = std.Io.Timestamp.now(io, .awake);
for (0..iters) |_| {
    fn(fn, params, out);
    std.mem.doNotOptimizeAway(out.ptr);
}
const ns = start.untilNow(io, .awake).nanoseconds;
```

### `@bitCast`

Yes, you deserve a whole section you slowpoke. Going straight from a vector of f32 to an array of bytes actually uses a for loop. Keeping the same element type is actually the fastest way to make it work well.

## Journal

### 18/10/2026

- Documented the project and its goals, as well as the tools and resources that I will use to implement it.
- Read the `safetensors` format specification and understood how to read the header and the tensor metadata.

### 19/10/2026

- Wrote formatConversion.zig, which contains the functions to dequantize the NVFP4 values into `f32` values.
- Started looking at SIMD and `@Vector` in Zig, to see how to implement the dequantization in a vectorized way.
- Started looking at how to benchmark the two implementations, and see if multicore is even worth it in the first place: what is the bottleneck? I/O or CPU? If I/O, then multicore won't help much. If CPU, then multicore could help a lot.
- Investigated the `std.Io` interface and how to use it to read the `.safetensors` file and write the dequantized values to a `std.Io.Writer`. This is cursed. This is Eldritch. The documentation was redacted by horrors from beyond.
- Wrote the code for reading the header length, the header

Questions:

- Should I allocate the reading block as a function of the matrices? or split it in multiple chunks?
- What should I do about the other format? Cast them?

### 20/10/2026

- Finished the structure reading of the json
- Allowed a json structure to map into a job of nvfp4 computations
- Reworked the main
- Struggled with the documentation and the ways of sorting/finding elements

What I can do next is write the code for dispatch:

- write to get the readers where i want them
- implement chunking
- implement backend
- dispatch the chunking on the right backend
- then SIMD
- then Multicore
- and then we're good?

### 21/10/2026

- Dedicated most of the time to figure out what the stream meant and to find a good stucture that felt good and made sense in my brain. I want to do a sort of map / reduce as in hadoop
- In fact, i'm going to serve a streaming tensor model so I actually first need to build a layout of what to process and how to process it to stream it afterwards
- So first, i do a pass on the structure, i gather jobs with distinct roles by tagging them. I rebuild a new header with the right sectors (this on hold for now). Then I stream depending on the job (either copy/dequantize)

### 22/10/2026

- Finished the job implementation, i don't know really how the streaming is going to be so I set some placeholder info i could get on the documentation and decided to focus on making sure i was labelling my jobs correctly. I got to the end of it, i think (until i figure out what i didn't do correctly for the stream)
- Started reading a lot about the format and simd to get to a satisfying code. Asked theo about how bit shifting and dequantization worked
- finished e2m1 (easy enough with the full explanation)

### 23/10/2026

- Finished E4M3 which was way more complicated than the other due to having more bits
- Will lead on streaming.

### 27/10/2026

- Made progress towards dequantize (need to compute the multiplication between scales and serve the results )
- Made progress towards multicore (i have a clear idea of what to do)

### 29/10/2026

- Figured out the low nibble and high nibble: Learned @shuffle to permute
- Learned about group, about atomic adds like in c++ about sending plates to the worker like I'm playing overcooked


## Open Questions

- Is multicore really worth it? Is the compute the bottleneck or the memory? Not really sure of doing it for the project, it's a long time investment for maybe a very small reward.

## What I'm not going to do

- Verify that the file is not corrupted
- Verify that the sectors are matching
