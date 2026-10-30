const std = @import("std");
const Io = std.Io;

const e2m1ToF32Simd = @import("e2m1.zig").e2m1ToF32Simd;
const e4m3ToF32bits = @import("e4m3.zig").e4m3ToF32bits;
// one e4m3 decode covers 16 scales: the padding in decodeGroup depends on it
const BLOCKS_PER_GROUP = 16;

// Receives weight buffer, scales buffer, scale 2, outbuffer
// 1. split blocks into K ranges, each a multiple of 16 blocks
// 2. Each blocks get sent to a worker
// 2.1  Decodes its 16 scales and apply the float
// 2.2. Compute the corresponding e2m1 by splitting the nibbles (&0x0F to mask the first, >> 4 to shift the last) then brought back in order
// 2.3. Decode to nvfp4, then write in the block for one given factor scale2*scale1*16valuese2m1
// 3. Write in the correct position of the buffer
// 4. wait for all workers to finish

// https://www.ziglang.in/learn/concurrency/async-future-group/
// https://www.ziglang.in/learn/concurrency/atomics/

// Everything the workers share for one mini-job: the three buffers, scale2, and the serving counter
const Plates = struct {
    weights: []const u8,
    scales: []const u8,
    scale2: f32,
    out: []u8,
    // a plate is a range of blocks, a multiple of 16 so each plate decodes whole groups of scales
    blocksPerPlate: usize,
    numPlates: usize,
    numBlocks: usize,
    // the next plate nobody has taken yet
    // actually a lock that dictates when a plate is taken
    next: std.atomic.Value(usize) = .init(0),
};

fn interleaveMask() @Vector(16, i32) {
    var m: [16]i32 = undefined;
    for (0..8) |i| {
        m[2 * i] = @intCast(i);
        m[2 * i + 1] = ~@as(i32, @intCast(i));
    }
    return m;
}

test "interleave vector" {
    // 8 times 33
    const a: @Vector(8, u8) = @splat(0x21);
    // 8 times 137
    const b: @Vector(8, u8) = @splat(0x89);
    const out: [16]u8 = @shuffle(u8, a, b, interleaveMask());
    // a, b, a, b
    for (out, 0..) |byte, i| {
        const want: u8 = if (i % 2 == 0) 0x21 else 0x89;
        try std.testing.expectEqual(want, byte);
    }
}

// This is the main function used to generate the nvfp4
// https://www.openmymind.net/Zigs-memcpy-copyForwards-and-copyBackwards/
// https://hive.blog/hive-196387/@scipio/learn-zig-series-19-simd-with-vector
fn decodeGroup(weights: []const u8, scales: []const u8, scale2: f32, out: []u8) void {
    // pad a short group with zeros so the decode always sees 16 scales to fit (the size of block)
    var padded: [BLOCKS_PER_GROUP]u8 = @splat(0);
    @memcpy(padded[0..scales.len], scales);
    const scale2s: @Vector(BLOCKS_PER_GROUP, f32) = @splat(scale2);
    const factors = e4m3ToF32bits(BLOCKS_PER_GROUP, padded) * scale2s;
    // a vector can't be indexed with a runtime k of course so i need a new vector for that
    const factorArray: [BLOCKS_PER_GROUP]f32 = factors;

    const blocks: []const [8]u8 = std.mem.bytesAsSlice([8]u8, weights);
    for (blocks, 0..) |block, k| {
        const bytes: @Vector(8, u8) = block;
        // first values in the low nibble, second values in the high nibble
        const low = bytes & @as(@Vector(8, u8), @splat(0x0F));
        const high = bytes >> @splat(4);
        // back in file order: low 0 high 0 low 1 high 1
        const codes: @Vector(16, u8) = @shuffle(u8, low, high, interleaveMask());
        const e2m1 = e2m1ToF32Simd(@truncate(codes));
        const factor: @Vector(16, f32) = @splat(factorArray[k]);
        const nvfp4: @Vector(64, u8) = @bitCast(factor * e2m1);
        // notation is: slice from position k*64 to out. Out of this, we have the remaining out buffer available which was preselected, so we slice just for the 64 bits that we're going to write, but it would be only a pointer there
        // we dereference the pointer to assign value
        out[k * 64 ..][0..64].* = nvfp4;
    }
}

// a plate is blocksPerPlate blocks, decoded as groups of 16 (one e4m3 decode each)
// only the last group of the last plate can be short, hence padded
fn decodePlate(weights: []const u8, scales: []const u8, scale2: f32, out: []u8) void {
    var groupStart: usize = 0;
    while (groupStart < scales.len) : (groupStart += BLOCKS_PER_GROUP) {
        const groupEnd = @min(groupStart + BLOCKS_PER_GROUP, scales.len);
        decodeGroup(weights[8 * groupStart .. 8 * groupEnd], scales[groupStart..groupEnd], scale2, out[64 * groupStart .. 64 * groupEnd]);
    }
}

fn worker(plates: *Plates) void {
    while (true) {
        // Increase the counter of blocks taken
        const k = plates.next.fetchAdd(1, .monotonic);
        // If we have finished distributing all the plates then we are done
        if (k >= plates.numPlates) return;
        // the plate is computed from its serving
        const start = k * plates.blocksPerPlate;
        // the end is the minimum between the number of blocks remaining to compute and the max theoritical
        const end = @min(start + plates.blocksPerPlate, plates.numBlocks);
        // 8 bytes of weights, 1 scale, 64 bytes of output per block that we decode in the meantime
        decodePlate(plates.weights[8 * start .. 8 * end], plates.scales[start..end], plates.scale2, plates.out[64 * start .. 64 * end]);
    }
}

// blocksPerPlate: work a worker takes per serving, big enough that the fetchAdd is negligible next to the decoding
// minBlocksForWorkers: below this number of blocks, waking the threads costs more than the decoding itself
// both set once in Dequantizer.init, check claude benchmark
pub fn bufferToWorker(io: Io, weights: []const u8, scales: []const u8, scale2: f32, out: []u8, numWorkers: usize, blocksPerPlate: usize, minBlocksForWorkers: usize) Io.Cancelable!void {
    // io.cancelable for the group.await
    // a plate of 0 blocks would divide by zero, Dequantizer.init rejects it
    std.debug.assert(blocksPerPlate > 0);
    var plates: Plates = .{
        .weights = weights,
        .scales = scales,
        .scale2 = scale2,
        .out = out,
        .blocksPerPlate = blocksPerPlate,
        // rounded up, so a short last plate still gets served
        .numPlates = (weights.len / 8 + blocksPerPlate - 1) / blocksPerPlate,
        .numBlocks = weights.len / 8,
    };

    // if the data is below the threshold of taking more memory access than compute, then we don't.
    if (numWorkers == 1 or plates.numBlocks < minBlocksForWorkers) {
        worker(&plates);
        return;
    }
    // call the group and dispatch the data
    var group: Io.Group = .init;
    for (0..numWorkers) |_| group.async(io, worker, .{&plates});
    try group.await(io);
}

test "buffers to workers" {
    const io = std.testing.io;
    const cpus = 12;
    // init a number of blocks (192 = 16x12)
    const blocks = 200;
    // 0x21: first nibble 0x1 = 0.5, second nibble 0x2 = 1.0
    const weights: [blocks * 8]u8 = @splat(0x21);
    // 0x40: E4M3 with e = 8, m = 0, so 2^(8-7) = 2.0
    const scales: [blocks]u8 = @splat(0x40);
    const scale2: f32 = 0.25;
    // get the values out as float32
    var out: [blocks * 64]u8 = undefined;
    try bufferToWorker(io, &weights, &scales, scale2, &out, cpus, 256, 8192);
    // ok now the test passes, but we need to look at the decoding
    // factor is 2*0.25 = 0.5 for the first bit, and 0.5*0.5 = 0.25 for the second
    // given the weights 0.5 and second 1.0 it should give
    // 0.25 yielding bytes 0, 0, 128, 62
    // 0.5 yielding bytes 0, 0, 0, 63
    // len should be 200x64
    try std.testing.expectEqual(out.len, 200 * 64);
    const high: [4]u8 = .{ 0, 0, 128, 62 };
    const low: [4]u8 = .{ 0, 0, 0, 63 };
    var k: usize = 0;
    while (k < out.len) : (k += 4) {
        const float: [4]u8 = out[k..][0..4].*;
        if (k % 8 == 0) {
            try std.testing.expectEqual(high, float);
        } else {
            try std.testing.expectEqual(low, float);
        }
    }
}
