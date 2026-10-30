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

    // Next jsonLen bytes: the JSON header itself.
    const json = r.readAllocAll(allocator, @intCast(jsonLen)) catch |err| switch (err) {
        error.ReadFailed => return fileReader.err.?,
        error.OutOfMemory => return error.OutOfMemory,
        error.EndOfStream => return error.InvalidSafetensors, // header truncated
    };

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

    // 5 tensors
    const map = structure.map.value.map;
    try std.testing.expectEqual(5, map.count());

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
