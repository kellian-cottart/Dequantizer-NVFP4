const std = @import("std");
const Io = std.Io;
const read = @import("Dequantizer").read;
const Dequantizer = @import("Dequantizer").dequantizer.Dequantizer;

/// I want to make an application much like in C++
/// Our output is a stream of the whole dequantized tensor.

// * Build the jobs (dispatch) :
//   1. If its a nvfp4 triplet, i mark it dequantize
//   2. if its a scale, skip
//   3. if its anything else, copy and hold the position of bytes
// * Sort the jobs by position
// * Build the output header of the job
// * Stream
//   1. Give the header + metadata
//   2. Perform the jobs sequentially
//      1. Copy: just hand out the bytes (can use mini-jobs later)
//      2. Dequantize (can use mini-jobs later):
//         1. Read the scales
//         2. Read the weights
//         3. Dequantize them into f32 and handle them out

// Parameters of the dequantizer, from the measurements in Claude Benchmark
// (Ryzen 5 7535U: 6 cores / 12 threads, 16 MB of L3, a single DDR5 stick)

// a mini-job of 64 * 1024 blocks writes 64 * 1024 * 64 bytes = 4 MB of f32, which stays in the 16 MB L3 (~4x); bigger mini-jobs spill to memory leading to only ~2x
//
const CHUNK_SIZE = 64 * 1024 * 8;

// init clamps it to the cpu count of the machine it runs on, will depend heavily on the machine used
const NUM_WORKERS = 12;

// blocks a worker takes per serving
// from 256 the fetchAdd is negligible next to the decoding, 1024 gains nothing more, multiple of 16
const BLOCKS_PER_PLATE = 256;

// below 16 * 1024 blocks (1 MB of f32) some worker counts were slower than one thread, making them useless, so decode on the calling thread
const MIN_BLOCKS_FOR_WORKERS = 16 * 1024;

// buffer size, big enough for the small tensors, but smaller than minijobs
const STDOUT_BUFFER_SIZE = 64 * 1024;

pub fn main(init: std.process.Init) !void {

    // Initialize the std.io interface
    const io = init.io;

    // Allocate memory
    const allocator = init.arena.allocator();

    // Define iterator for the args
    var it = try init.minimal.args.iterateAllocator(allocator);
    defer it.deinit();

    // Arg sys
    var filename: ?[]const u8 = null;
    _ = it.next(); // argv[0] is the program's own path
    while (true) {
        // Only advance the iterator here at the beginning of a fresh check
        const arg = it.next() orelse break;
        if (std.mem.eql(u8, arg, "--filename") or std.mem.eql(u8, arg, "-f")) {
            filename = it.next() orelse return error.FileError;
        }
    }

    // Stop before opening anything if the flag was never given
    const inputFilename = filename orelse {
        std.debug.print("./zig-out/bin/Dequantizer.exe -f data/model.safetensors > data/out.safetensors\n", .{});
        std.process.exit(1);
    };
    // for windows
    // cmd /c "zig-out\bin\Dequantizer.exe -f data\model.safetensors > data\out.safetensors

    // Open the file
    const file = try Io.Dir.cwd().openFile(io, inputFilename, .{});
    defer file.close(io);

    // The dequantizer reads the structure, makes the jobs and builds the header
    var dequantizer: Dequantizer = try .init(allocator, io, file, CHUNK_SIZE, NUM_WORKERS, BLOCKS_PER_PLATE, MIN_BLOCKS_FOR_WORKERS);
    defer dequantizer.deinit();

    // The output is a buffered writer on stdout
    var out_buf: [STDOUT_BUFFER_SIZE]u8 = undefined;
    var stdout_writer = Io.File.stdout().writer(io, &out_buf);

    // Stream the information to stdout
    _ = try dequantizer.interface.streamRemaining(&stdout_writer.interface);
    try stdout_writer.interface.flush();
}
