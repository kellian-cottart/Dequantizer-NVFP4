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

pub fn main(init: std.process.Init) !void {

    // Initialize the std.io interface
    const io = init.io;

    // Initialize shared memory
    const allocator = init.arena.allocator();
    const filename = "src/testdata/tiny_nvfp4.safetensors";

    // Open the file
    const file = try Io.Dir.cwd().openFile(io, filename, .{});
    defer file.close(io);

    // The dequantizer reads the structure, makes the jobs and builds the header
    // 256 * 1024 is adjustable
    var dequantizer: Dequantizer = try .init(allocator, io, file, 256 * 1024);
    defer dequantizer.deinit();

    // The output is a buffered writer on stdout
    var out_buf: [64 * 1024]u8 = undefined;
    var stdout_writer = Io.File.stdout().writer(io, &out_buf);

    // Stream the information to stdout
    _ = try dequantizer.interface.streamRemaining(&stdout_writer.interface);
    try stdout_writer.interface.flush();
}
