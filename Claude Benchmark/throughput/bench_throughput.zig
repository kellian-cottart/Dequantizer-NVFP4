// Throughput of bufferToWorker (src/backend/simd/multiply.zig) for 1 to 12 workers, at sizes around the
// best working spot. Calls only bufferToWorker, with the plate size and threshold of main.zig.
//
// Run from the repository root (throughput.ipynb does it if throughput.csv is missing):
//   zig run -OReleaseFast -mcpu=native --dep dequantizer "-Mroot=Claude Benchmark/throughput/bench_throughput.zig" -Mdequantizer=src/root.zig

const std = @import("std");
const Io = std.Io;
const bufferToWorker = @import("dequantizer").backend.simd.multiply.bufferToWorker;

const outPath = "Claude Benchmark/throughput/throughput.csv";
const blocksPerPlate = 256;
const minBlocksForWorkers = 16 * 1024;
// 16384 ... 262144 blocks per call: 1 MB to 16 MB of f32 output
const sizes = [_]usize{ 16384, 32768, 65536, 131072, 262144 };
const maxWorkers = 12;
const repetitions = 20;
// each timing repeats the call until about this many blocks were decoded
const blocksPerTiming = 262144;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    const cpus = std.Thread.getCpuCount() catch 1;

    const maxBlocks = sizes[sizes.len - 1];
    // every block decodes to 0.25, 0.5, ...
    const weights = try gpa.alloc(u8, maxBlocks * 8);
    defer gpa.free(weights);
    @memset(weights, 0x21);
    const scales = try gpa.alloc(u8, maxBlocks);
    defer gpa.free(scales);
    @memset(scales, 0x40);
    const out = try gpa.alloc(u8, maxBlocks * 64);
    defer gpa.free(out);

    var csv: Io.Writer.Allocating = .init(gpa);
    defer csv.deinit();
    try csv.writer.print("blocks,workers,repetition,ns_per_call,cpus\n", .{});

    // warm the thread pool so the first group does not pay for spawning threads
    try bufferToWorker(io, weights, scales, 0.25, out, cpus, blocksPerPlate, minBlocksForWorkers);

    for (0..repetitions) |repetition| {
        for (sizes) |blocks| {
            const calls = @max(1, blocksPerTiming / blocks);
            for (1..maxWorkers + 1) |numWorkers| {
                const start = Io.Timestamp.now(io, .awake);
                for (0..calls) |_| try bufferToWorker(io, weights[0 .. blocks * 8], scales[0..blocks], 0.25, out[0 .. blocks * 64], numWorkers, blocksPerPlate, minBlocksForWorkers);
                const stop = Io.Timestamp.now(io, .awake);
                const elapsed: f64 = @floatFromInt(stop.nanoseconds - start.nanoseconds);
                try csv.writer.print("{d},{d},{d},{d:.1},{d}\n", .{ blocks, numWorkers, repetition, elapsed / @as(f64, @floatFromInt(calls)), cpus });
            }
        }
        std.debug.print("repetition {d}/{d}\n", .{ repetition + 1, repetitions });
    }

    // the decode must still be right, or the timings measure nothing
    for (std.mem.bytesAsSlice(f32, out), 0..) |value, index| {
        const want: f32 = if (index % 2 == 0) 0.25 else 0.5;
        if (value != want) return error.WrongDecode;
    }
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = outPath, .data = csv.written() });
    std.debug.print("wrote {s}\n", .{outPath});
}
