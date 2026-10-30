const std = @import("std");
const Io = std.Io;
const read = @import("../read.zig");
const Job = @import("jobs.zig").Job;
const makeJobs = @import("jobs.zig").makeJobs;

// Build the header of the output file: 8 bytes of json length, then the json
// The jobs are already sorted by input offset, so the output tensors go one after the other in that order
// and the data offsets are rewritten while iterating
// copy: same dtype, shape and size
// dequantize: F32, the last dimension doubles, we eat the offsets from scales and multiply by 4 the u8
// The scales are not in the list of jobs, so they disappear from the header
pub fn buildHeader(allocator: std.mem.Allocator, jobs: []const Job, metadata: read.Metadata) ![]u8 {
    var json: Io.Writer.Allocating = .init(allocator);
    defer json.deinit();
    var stringify: std.json.Stringify = .{ .writer = &json.writer };

    try stringify.beginObject();
    if (metadata.__metadata__.map.count() > 0) {
        try stringify.objectField("__metadata__");
        try stringify.write(metadata.__metadata__);
    }

    // where the next tensor starts in the output data
    var offset: u64 = 0;
    // the new shape of a dequantized tensor
    var shapeBuffer: [8]u64 = undefined;
    for (jobs) |job| {
        var info: read.TensorInfo = undefined;
        switch (job.action) {
            .copy => |copyInfo| {
                // same tensor, only moved
                info = copyInfo;
                info.data_offsets = .{ offset, offset + (copyInfo.data_offsets[1] - copyInfo.data_offsets[0]) };
            },
            .dequantize => |nvfp4| {
                const weight = nvfp4.weight;
                if (weight.shape.len == 0 or weight.shape.len > shapeBuffer.len) return error.InvalidShape;
                const shape = shapeBuffer[0..weight.shape.len];
                // just copy the memory to it
                @memcpy(shape, weight.shape);
                // shape counts values, there are twice more values now that we unpacked
                // the low and upper nibbles
                shape[shape.len - 1] *= 2;
                // here we have 8 times more memory due to going u8 to f32, for each two values
                const size = (weight.data_offsets[1] - weight.data_offsets[0]) * 8;
                info = .{ .dtype = .F32, .shape = shape, .data_offsets = .{ offset, offset + size } };
            },
        }
        offset = info.data_offsets[1];
        try stringify.objectField(job.name);
        try stringify.write(info);
    }
    try stringify.endObject();

    // pad with spaces so the data after the header starts on a multiple of 8,
    // like the safetensors library does
    // prepare in safetensors/src/tensor.rs forces alignement to 8 bytes
    while (json.written().len % 8 != 0) try json.writer.writeByte(' ');

    const header = try allocator.alloc(u8, 8 + json.written().len);
    std.mem.writeInt(u64, header[0..8], json.written().len, .little);
    @memcpy(header[8..], json.written());
    return header;
}

test "build the header" {
    const allocator = std.testing.allocator;
    // the tensors of the test file
    const jsonStr =
        \\{
        \\  "__metadata__":        { "format": "pt" },
        \\  "tiny.input_scale":    { "dtype": "F32",     "shape": [],      "data_offsets": [0, 4] },
        \\  "tiny.weight_scale_2": { "dtype": "F32",     "shape": [],      "data_offsets": [4, 8] },
        \\  "tiny.norm.weight":    { "dtype": "BF16",    "shape": [8],     "data_offsets": [8, 24] },
        \\  "tiny.weight_scale":   { "dtype": "F8_E4M3", "shape": [2, 2],  "data_offsets": [24, 28] },
        \\  "tiny.weight":         { "dtype": "U8",      "shape": [2, 16], "data_offsets": [28, 60] }
        \\}
    ;
    var parsedJson = try read.parseHeader(allocator, jsonStr);
    defer parsedJson.deinit();
    _ = parsedJson.value.map.orderedRemove("__metadata__");
    const metadata = try read.parseMetadata(allocator, jsonStr);
    defer metadata.deinit();

    const jobs = try makeJobs(allocator, &parsedJson.value.map);
    defer allocator.free(jobs);

    const header = try buildHeader(allocator, jobs, metadata.value);
    defer allocator.free(header);

    // 8 bytes of length, then the json padded to a multiple of 8
    const jsonLength = std.mem.readInt(u64, header[0..8], .little);
    try std.testing.expectEqual(header.len - 8, jsonLength);
    try std.testing.expectEqual(0, jsonLength % 8);

    // read it with our own parser
    var output = try read.parseHeader(allocator, header[8..]);
    defer output.deinit();
    const tensors = &output.value.map;
    // __metadata__, the norm and the weight: the scales are gone
    try std.testing.expectEqual(3, tensors.count());

    //  same dtype and shape, moved to the start of the data, we copy it
    const norm = tensors.get("tiny.norm.weight") orelse return error.MissingTensor;
    try std.testing.expectEqual(read.Dtype.BF16, norm.dtype);
    try std.testing.expectEqualSlices(u64, &.{8}, norm.shape);
    try std.testing.expectEqual([2]u64{ 0, 16 }, norm.data_offsets);

    // F32, [2, 16] packed bytes become [2, 32] values, 32 bytes become 256, we dequantize it
    const weight = tensors.get("tiny.weight") orelse return error.MissingTensor;
    try std.testing.expectEqual(read.Dtype.F32, weight.dtype);
    try std.testing.expectEqualSlices(u64, &.{ 2, 32 }, weight.shape);
    try std.testing.expectEqual([2]u64{ 16, 272 }, weight.data_offsets);

    // the metadata is kept
    const outputMetadata = try read.parseMetadata(allocator, header[8..]);
    defer outputMetadata.deinit();
    try std.testing.expectEqualStrings("pt", outputMetadata.value.__metadata__.map.get("format") orelse return error.MissingMetadata);
}
