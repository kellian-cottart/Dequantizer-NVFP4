const std = @import("std");
const Io = std.Io;

// These are the data types supported by the safetensors format.
pub const Dtype = enum { BOOL, U8, I8, F8_E5M2, F8_E4M3, I16, U16, F16, BF16, I32, U32, F32, F64, I64, U64 };

// This is the information about a tensor that is stored in the safetensors format.
pub const TensorInfo = struct {
    dtype: ?Dtype = null,
    shape: []const u64 = &.{},
    data_offsets: [2]u64 = .{ 0, 0 },
};

// This is the information about the NVFP4 that has associated typing
pub const Nvfp4Tensor = struct {
    weight: TensorInfo, // U8,
    scale: TensorInfo, // F8_E4M3,
    scale_2: TensorInfo, // F32,
};

// Reading the json header of the safetensors file, the reader is then at the start of the tensors data
// We take the file reader and not the interface, because the real error is stored in the file reader
pub fn read_header_json(allocator: std.mem.Allocator, file_reader: *Io.File.Reader) ![]u8 {
    // Create a reader interface to read from the file
    const r: *Io.Reader = &file_reader.interface;

    // First 8 bytes: little-endian u64, the length of the JSON header in bytes.
    const json_len = r.takeInt(u64, .little) catch |err| switch (err) {
        error.ReadFailed => return file_reader.err.?,
        error.EndOfStream => return error.InvalidSafetensors, // file shorter than 8 bytes
    };
    std.debug.print("JSON header length: {}\n", .{json_len});

    // Next json_len bytes: the JSON header itself.
    const json = r.readAllocAll(allocator, @intCast(json_len)) catch |err| switch (err) {
        error.ReadFailed => return file_reader.err.?,
        error.OutOfMemory => return error.OutOfMemory,
        error.EndOfStream => return error.InvalidSafetensors, // header truncated
    };
    std.debug.print("JSON header: {s}\n", .{json});

    return json;
}

test "read the json header of a safetensors file" {
    const io = std.testing.io;
    const heap_allocator = std.testing.allocator;

    // Open the test file, and making sure it is closed when we leave this scope
    const file = try Io.Dir.cwd().openFile(io, "src/testdata/tiny_nvfp4.safetensors", .{});
    defer file.close(io);

    // Create a reader to read from the file
    var read_buf: [1024]u8 = undefined;
    var file_reader = file.reader(io, &read_buf);

    // Reading the json header, and making sure it is freed when we leave this scope
    const json = try read_header_json(heap_allocator, &file_reader);
    defer heap_allocator.free(json);

    // checking if the header has the expected length, and is a json object
    try std.testing.expectEqual(384, json.len);
    try std.testing.expect(std.mem.startsWith(u8, json, "{"));
}

// Parse the json header into a map of tensor infos
pub fn parse_header(allocator: std.mem.Allocator, json: []const u8) !std.json.Parsed(std.json.ArrayHashMap(TensorInfo)) {
    // Allocating the json reader and the io reader, and making sure they are deinitialized when we leave this scope
    var json_io: Io.Reader = .fixed(json);
    var json_reader: std.json.Reader = .init(allocator, &json_io);
    defer json_reader.deinit();

    // Parsing the json into a map of tensor infos
    return std.json.parseFromTokenSource(std.json.ArrayHashMap(TensorInfo), allocator, &json_reader, .{ .ignore_unknown_fields = true });
}

test "parse a json file" {
    const json_str =
        \\{
        \\    "tensor1": { "dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24] },
        \\    "tensor2": { "dtype": "I16", "shape": [4, 5], "data_offsets": [24, 64] }
        \\}
    ;
    const heap_allocator = std.testing.allocator;

    // Parsing the json into a map of tensor infos, and making sure it is deinitialized when we leave this scope
    const parsed = try parse_header(heap_allocator, json_str);
    defer parsed.deinit();

    // Checking if there are two fields in the map
    const map = parsed.value.map;
    try std.testing.expectEqual(2, map.count());

    // checking if the first tensor has the expected dtype, shape, and data offsets
    const t1 = map.get("tensor1") orelse return error.MissingTensor;
    try std.testing.expectEqual(Dtype.F32, t1.dtype);
    try std.testing.expectEqualSlices(u64, &.{ 2, 3 }, t1.shape);
    try std.testing.expectEqual([2]u64{ 0, 24 }, t1.data_offsets);
}

pub fn by_weight_offset(_: void, a: Nvfp4Tensor, b: Nvfp4Tensor) bool {
    return a.weight.data_offsets[0] < b.weight.data_offsets[0];
}

test "sort list by weight offset" {
    var tensors = [_]Nvfp4Tensor{
        .{ .weight = .{ .data_offsets = .{ 130, 146 } }, .scale = .{}, .scale_2 = .{} },
        .{ .weight = .{ .data_offsets = .{ 50, 114 } }, .scale = .{}, .scale_2 = .{} },
        .{ .weight = .{ .data_offsets = .{ 114, 130 } }, .scale = .{}, .scale_2 = .{} },
    };
    std.mem.sort(Nvfp4Tensor, &tensors, {}, by_weight_offset);
    try std.testing.expectEqual(50, tensors[0].weight.data_offsets[0]);
    try std.testing.expectEqual(114, tensors[1].weight.data_offsets[0]);
    try std.testing.expectEqual(130, tensors[2].weight.data_offsets[0]);
}

pub fn find_nvp4(allocator: std.mem.Allocator, map: *const std.StringArrayHashMapUnmanaged(TensorInfo), name: []const u8) !?Nvfp4Tensor {
    // If there's no weight, we return null
    if (!std.mem.endsWith(u8, name, ".weight")) return null;
    const weight = map.get(name) orelse return null;

    // Else there's weight, and we try to get the other field
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return null;
    const prefix: []const u8 = name[0..dot];

    // Allocate dynamically a string
    // Malloc all over again
    const scale_name = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{prefix});
    defer allocator.free(scale_name);
    const scale = map.get(scale_name) orelse return null;
    // Same
    const scale_name_2 = try std.fmt.allocPrint(allocator, "{s}.weight_scale_2", .{prefix});
    defer allocator.free(scale_name_2);
    const scale_2 = map.get(scale_name_2) orelse return null;

    // Check the shape is a two sized vector from start to finish
    if (weight.shape.len != 2 or scale.shape.len != 2) return null;
    // Check that for the first scale we do really have 8 bits
    if (scale.shape[0] != weight.shape[0] or scale.shape[1] * 8 != weight.shape[1]) return null;
    // Check type
    if (weight.dtype != .U8 or scale.dtype != .F8_E4M3 or scale_2.dtype != .F32) return null;
    return .{ .weight = weight, .scale = scale, .scale_2 = scale_2 };
}

test "find nvfp4" {
    const json_str =
        \\{
        \\  "__metadata__": { "format": "pt" },
        \\  "a.input_scale":    { "dtype": "F32",     "shape": [],      "data_offsets": [0, 4] },
        \\  "a.weight_scale_2": { "dtype": "F32",     "shape": [],      "data_offsets": [4, 8] },
        \\  "c.weight_scale_2": { "dtype": "F32",     "shape": [],      "data_offsets": [8, 12] },
        \\  "d.weight_scale_2": { "dtype": "F32",     "shape": [],      "data_offsets": [12, 16] },
        \\  "d.weight_scale":   { "dtype": "BF16",    "shape": [2, 1],  "data_offsets": [16, 20] },
        \\  "norm.weight":      { "dtype": "BF16",    "shape": [8],     "data_offsets": [20, 36] },
        \\  "a.weight_scale":   { "dtype": "F8_E4M3", "shape": [4, 2],  "data_offsets": [36, 44] },
        \\  "b.weight_scale":   { "dtype": "F8_E4M3", "shape": [2, 1],  "data_offsets": [44, 46] },
        \\  "c.weight_scale":   { "dtype": "F8_E4M3", "shape": [2, 2],  "data_offsets": [46, 50] },
        \\  "a.weight":         { "dtype": "U8",      "shape": [4, 16], "data_offsets": [50, 114] },
        \\  "b.weight":         { "dtype": "U8",      "shape": [2, 8],  "data_offsets": [114, 130] },
        \\  "c.weight":         { "dtype": "U8",      "shape": [2, 8],  "data_offsets": [130, 146] },
        \\  "d.weight":         { "dtype": "U8",      "shape": [2, 8],  "data_offsets": [146, 162] }
        \\}
    ;
    const heap_allocator = std.testing.allocator;

    // Creating the list to hold the NVFP4 Tensors
    var list: std.ArrayList(Nvfp4Tensor) = .empty;
    defer list.deinit(heap_allocator);

    // Creating the parser
    const parsed_json = try parse_header(heap_allocator, json_str);
    defer parsed_json.deinit();

    const map = &parsed_json.value.map;

    // We iterate on the keys
    // We pick up the name, if it has weight,
    // we create a nvfp4tensor that we add to the list
    for (map.keys()) |key| {
        const out = try find_nvp4(heap_allocator, map, key) orelse continue;
        try list.append(heap_allocator, out);
    }
    try std.testing.expectEqual(1, list.items.len);
    try std.testing.expectEqual([2]u64{ 50, 114 }, list.items[0].weight.data_offsets);
}

// Gather every NVFP4 tensor of the header into list
pub fn collect_nvfp4(allocator: std.mem.Allocator, map: *const std.StringArrayHashMapUnmanaged(TensorInfo), list: *std.ArrayList(Nvfp4Tensor)) !void {
    for (map.keys()) |key| {
        const t = try find_nvp4(allocator, map, key) orelse continue;
        try list.append(allocator, t);
    }
}
