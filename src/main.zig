const std = @import("std");
const Io = std.Io;
const read = @import("Dequantizer").read;
const dispatch = @import("Dequantizer").dispatch;

/// Writing a little writer to read a file and print its contents to stdout, using the new std.Io interface.
/// https://zig.guide/standard-library/readers-and-writers/ (for 0.15.2)
pub fn main(init: std.process.Init) !void {

    // Initialize the std.io interface
    const io = init.io;

    // Initialize shared memory
    const allocator = init.arena.allocator();

    const filename = "src/testdata/tiny_nvfp4.safetensors";

    // Open the file here, the tensors will be read from it later
    const file = try Io.Dir.cwd().openFile(io, filename, .{});
    defer file.close(io);

    // Get the structure of the file
    const parsed = try get_structure(allocator, io, file);
    defer parsed.deinit();

    // Create the jobList
    var jobList: std.ArrayList(read.Nvfp4Tensor) = .empty;
    defer jobList.deinit(allocator);

    // Collect the jobs
    try read.collect_nvfp4(allocator, &parsed.value.map, &jobList);

    // We have the job list!
    std.debug.print("\n{any}\n", .{jobList.items[0]});

    // now we must give the jobs
}

pub fn get_structure(allocator: std.mem.Allocator, io: Io, file: Io.File) !std.json.Parsed(std.json.ArrayHashMap(read.TensorInfo)) {
    // Create a reader to read the header from the file
    var read_buf: [64 * 1024]u8 = undefined;
    var file_reader = file.reader(io, &read_buf);

    // Read the json header (length + header itself), then parse it
    const json = try read.read_header_json(allocator, &file_reader);
    return read.parse_header(allocator, json);
}
