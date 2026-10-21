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
    scale2: TensorInfo, // F32,
};

// Metadata entry of the reader to reconstruct the full file afterwards
pub const Metadata = struct {
    __metadata__: std.json.ArrayHashMap([]const u8) = .{},
};

pub const Structure = struct {
    metadata: std.json.Parsed(Metadata),
    map: std.json.Parsed(std.json.ArrayHashMap(TensorInfo)),
    // Where the tensor data starts in the file: data_offsets are relative to this
    dataStart: u64,

    // free the memory of the struct
    pub fn deinit(self: Structure) void {
        self.metadata.deinit();
        self.map.deinit();
    }
};

// Reading the json header of the safetensors file, the reader is then at the start of the tensors data
// We take the file reader and not the interface, because the real error is stored in the file reader
pub fn readHeaderJson(allocator: std.mem.Allocator, fileReader: *Io.File.Reader) ![]u8 {
    // Create a reader interface to read from the file
    const r: *Io.Reader = &fileReader.interface;

    // First 8 bytes: little-endian u64, the length of the JSON header in bytes.
    const jsonLen = r.takeInt(u64, .little) catch |err| switch (err) {
        error.ReadFailed => return fileReader.err.?,
        error.EndOfStream => return error.InvalidSafetensors, // file shorter than 8 bytes
    };
    std.debug.print("JSON header length: {}\n", .{jsonLen});

    // Next jsonLen bytes: the JSON header itself.
    const json = r.readAllocAll(allocator, @intCast(jsonLen)) catch |err| switch (err) {
        error.ReadFailed => return fileReader.err.?,
        error.OutOfMemory => return error.OutOfMemory,
        error.EndOfStream => return error.InvalidSafetensors, // header truncated
    };
    std.debug.print("JSON header: {s}\n", .{json});

    return json;
}

test "read the json header of a safetensors file" {
    const io = std.testing.io;
    const heapAllocator = std.testing.allocator;

    // Open the test file, and making sure it is closed when we leave this scope
    const file = try Io.Dir.cwd().openFile(io, "src/testdata/tiny_nvfp4.safetensors", .{});
    defer file.close(io);

    // Create a reader to read from the file
    var readBuf: [1024]u8 = undefined;
    var fileReader = file.reader(io, &readBuf);

    // Reading the json header, and making sure it is freed when we leave this scope
    const json = try readHeaderJson(heapAllocator, &fileReader);
    defer heapAllocator.free(json);

    // checking if the header has the expected length, and is a json object
    try std.testing.expectEqual(384, json.len);
    try std.testing.expect(std.mem.startsWith(u8, json, "{"));
}

// Parse the json header into a map of tensor infos
pub fn parseHeader(allocator: std.mem.Allocator, json: []const u8) !std.json.Parsed(std.json.ArrayHashMap(TensorInfo)) {
    // Allocating the json reader and the io reader, and making sure they are deinitialized when we leave this scope
    var jsonIo: Io.Reader = .fixed(json);
    var jsonReader: std.json.Reader = .init(allocator, &jsonIo);
    defer jsonReader.deinit();

    // Parsing the json into a map of tensor infos
    return std.json.parseFromTokenSource(std.json.ArrayHashMap(TensorInfo), allocator, &jsonReader, .{ .ignore_unknown_fields = true });
}

// Parse only the __metadata__ of the json header, the tensors are ignored
pub fn parseMetadata(allocator: std.mem.Allocator, json: []const u8) !std.json.Parsed(Metadata) {
    var jsonIo: Io.Reader = .fixed(json);
    var jsonReader: std.json.Reader = .init(allocator, &jsonIo);
    defer jsonReader.deinit();

    return std.json.parseFromTokenSource(Metadata, allocator, &jsonReader, .{ .ignore_unknown_fields = true });
}

test "parse a json file" {
    const jsonStr =
        \\{
        \\    "tensor1": { "dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24] },
        \\    "tensor2": { "dtype": "I16", "shape": [4, 5], "data_offsets": [24, 64] }
        \\}
    ;
    const heapAllocator = std.testing.allocator;

    // Parsing the json into a map of tensor infos, and making sure it is deinitialized when we leave this scope
    const parsed = try parseHeader(heapAllocator, jsonStr);
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

pub fn byWeightOffset(_: void, a: Nvfp4Tensor, b: Nvfp4Tensor) bool {
    return a.weight.data_offsets[0] < b.weight.data_offsets[0];
}

test "sort list by weight offset" {
    var tensors = [_]Nvfp4Tensor{
        .{ .weight = .{ .data_offsets = .{ 130, 146 } }, .scale = .{}, .scale2 = .{} },
        .{ .weight = .{ .data_offsets = .{ 50, 114 } }, .scale = .{}, .scale2 = .{} },
        .{ .weight = .{ .data_offsets = .{ 114, 130 } }, .scale = .{}, .scale2 = .{} },
    };
    std.mem.sort(Nvfp4Tensor, &tensors, {}, byWeightOffset);
    try std.testing.expectEqual(50, tensors[0].weight.data_offsets[0]);
    try std.testing.expectEqual(114, tensors[1].weight.data_offsets[0]);
    try std.testing.expectEqual(130, tensors[2].weight.data_offsets[0]);
}

pub fn findNvfp4(allocator: std.mem.Allocator, map: *const std.StringArrayHashMapUnmanaged(TensorInfo), name: []const u8) !?Nvfp4Tensor {
    // If there's no weight, we return null
    if (!std.mem.endsWith(u8, name, ".weight")) return null;
    const weight = map.get(name) orelse return null;

    // Else there's weight, and we try to get the other field
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return null;
    const prefix: []const u8 = name[0..dot];

    // Allocate dynamically a string
    // Malloc all over again
    const scaleName = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{prefix});
    defer allocator.free(scaleName);
    const scale = map.get(scaleName) orelse return null;
    // Same
    const scaleName2 = try std.fmt.allocPrint(allocator, "{s}.weight_scale_2", .{prefix});
    defer allocator.free(scaleName2);
    const scale2 = map.get(scaleName2) orelse return null;

    // Check the shape is a two sized vector from start to finish
    if (weight.shape.len != 2 or scale.shape.len != 2) return null;
    // Check that for the first scale we do really have 8 bits
    if (scale.shape[0] != weight.shape[0] or scale.shape[1] * 8 != weight.shape[1]) return null;
    // Check type
    if (weight.dtype != .U8 or scale.dtype != .F8_E4M3 or scale2.dtype != .F32) return null;
    return .{ .weight = weight, .scale = scale, .scale2 = scale2 };
}

test "find nvfp4" {
    const jsonStr =
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
    const heapAllocator = std.testing.allocator;

    // Creating the list to hold the NVFP4 Tensors
    var list: std.ArrayList(Nvfp4Tensor) = .empty;
    defer list.deinit(heapAllocator);

    // Creating the parser
    const parsedJson = try parseHeader(heapAllocator, jsonStr);
    defer parsedJson.deinit();

    const map = &parsedJson.value.map;

    // We iterate on the keys
    // We pick up the name, if it has weight,
    // we create a nvfp4tensor that we add to the list
    for (map.keys()) |key| {
        const out = try findNvfp4(heapAllocator, map, key) orelse continue;
        try list.append(heapAllocator, out);
    }
    try std.testing.expectEqual(1, list.items.len);
    try std.testing.expectEqual([2]u64{ 50, 114 }, list.items[0].weight.data_offsets);
}

// Gather every NVFP4 tensor of the header into list
pub fn collectNvfp4(allocator: std.mem.Allocator, map: *const std.StringArrayHashMapUnmanaged(TensorInfo), list: *std.ArrayList(Nvfp4Tensor)) !void {
    for (map.keys()) |key| {
        const t = try findNvfp4(allocator, map, key) orelse continue;
        try list.append(allocator, t);
    }
}

pub fn getStructure(allocator: std.mem.Allocator, io: Io, file: Io.File) !Structure {
    // Create a reader to read the header from the file
    var readBuf: [64 * 1024]u8 = undefined;
    var fileReader = file.reader(io, &readBuf);
    // Read the json header (length + header itself), then parse it
    const json = try readHeaderJson(allocator, &fileReader);
    // The parser copies every string it keeps, so the json can be freed once parsed
    defer allocator.free(json);

    // Defer the allocation in case of error
    const metadata = try parseMetadata(allocator, json);
    errdefer metadata.deinit();

    // Remove metadata from the map
    var map = try parseHeader(allocator, json);
    errdefer map.deinit();
    _ = map.value.map.orderedRemove("__metadata__");

    return .{ .metadata = metadata, .map = map, .dataStart = 8 + json.len };
}

test "get structure" {
    const io = std.testing.io;
    const heapAllocator = std.testing.allocator;

    // Open the test file, and making sure it is closed when we leave this scope
    const file = try Io.Dir.cwd().openFile(io, "src/testdata/tiny_nvfp4.safetensors", .{});
    defer file.close(io);

    // Getting the structure, and making sure it is deinitialized when we leave this scope
    const structure = try getStructure(heapAllocator, io, file);
    defer structure.deinit();

    // The data starts after the 8 bytes of length and the 384 bytes of json
    try std.testing.expectEqual(8 + 384, structure.dataStart);

    // The metadata only holds the format
    const metadata = structure.metadata.value.__metadata__.map;
    try std.testing.expectEqual(1, metadata.count());
    try std.testing.expectEqualStrings("pt", metadata.get("format") orelse return error.MissingMetadata);

    // 5 tensors, plus __metadata__ which is parsed as a TensorInfo without dtype
    const map = structure.map.value.map;
    try std.testing.expectEqual(6, map.count());

    // Checking the three tensors of the NVFP4
    const weight = map.get("tiny.weight") orelse return error.MissingTensor;
    try std.testing.expectEqual(Dtype.U8, weight.dtype);
    try std.testing.expectEqual([2]u64{ 28, 60 }, weight.data_offsets);
    const scale = map.get("tiny.weight_scale") orelse return error.MissingTensor;
    try std.testing.expectEqual(Dtype.F8_E4M3, scale.dtype);
    try std.testing.expectEqual([2]u64{ 24, 28 }, scale.data_offsets);
    const scale2 = map.get("tiny.weight_scale_2") orelse return error.MissingTensor;
    try std.testing.expectEqual(Dtype.F32, scale2.dtype);
    try std.testing.expectEqual([2]u64{ 4, 8 }, scale2.data_offsets);

    // The last tensor ends exactly at the end of the file (452 bytes), so dataStart is right
    try std.testing.expectEqual(452, structure.dataStart + weight.data_offsets[1]);
}
