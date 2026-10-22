const std = @import("std");
const read = @import("../read.zig");
const header = @import("header.zig");
const stream = @import("stream.zig");

// This is the information about the NVFP4 that has associated typing, we won't need the scales after dequantization
pub const Nvfp4Tensor = struct {
    weight: read.TensorInfo, // U8,
    scale: read.TensorInfo, // F8_E4M3,
    scale2: read.TensorInfo, // F32,
};

// Make so that we split into either a copy or dequantize job
// As their nature is different, they will hold different
pub const Action = union(enum) {
    copy: read.TensorInfo, // the tensor as it sits in the file
    dequantize: Nvfp4Tensor, // weight + scale + scale2
};

pub const Job = struct {
    name: []const u8, // key in the map, lives as long as structure
    action: Action,

    pub fn inputOffset(self: Job) u64 {
        return switch (self.action) {
            .copy => |info| info.data_offsets[0],
            .dequantize => |nvfp4| nvfp4.weight.data_offsets[0],
        };
    }

    pub fn byInputOffset(_: void, a: Job, b: Job) bool {
        return a.inputOffset() < b.inputOffset();
    }
};

//Tag every tensor (copy / dequantize / skip), drop the skips, sort by position
pub fn makeJobs(allocator: std.mem.Allocator, map: *const std.StringArrayHashMapUnmanaged(read.TensorInfo)) ![]Job {
    var jobs: std.ArrayList(Job) = .empty;
    errdefer jobs.deinit(allocator);

    for (map.keys(), map.values()) |key, info| {
        const newJob: Job = try createJob(allocator, map, key, info) orelse continue; // skip
        try jobs.append(allocator, newJob);
    }
    // Sort the jobs by input offset
    std.mem.sort(Job, jobs.items, {}, Job.byInputOffset);

    // Shrinks the memory allocation to the length of items in the array
    // In place or wherever a contiguous block is free
    return jobs.toOwnedSlice(allocator);
}

// Function tag: Receive a name, evaluate the type of job action and return it.
// Logic: 1. we try to see if its a nvfp4 job 2. if not then it goes to checking if its a scale, if it is, it is skipped 3. return the proper action for this job.

pub fn createJob(allocator: std.mem.Allocator, map: *const std.StringArrayHashMapUnmanaged(read.TensorInfo), key: []const u8, info: read.TensorInfo) !?Job {
    // if we have a NVFP4
    if (try findNvfp4(allocator, map, key)) |nvfp4| {
        return .{ .name = key, .action = .{ .dequantize = nvfp4 } };
    }
    // if it's an element of a valid NVFP4 group, or an input scale, then we skip it
    if (try isSkipped(allocator, map, key)) return null;
    return .{ .name = key, .action = .{ .copy = info } };
}

test "create a job" {
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

    // Creating the parser
    var parsedJson = try read.parseHeader(heapAllocator, jsonStr);
    defer parsedJson.deinit();

    // parseHeader keeps __metadata__ as a tensor with default fields, getStructure removes it, so we do the same
    _ = parsedJson.value.map.orderedRemove("__metadata__");
    const map = &parsedJson.value.map;

    // Creating the list to hold the jobs
    var jobList: std.ArrayList(Job) = .empty;
    defer jobList.deinit(heapAllocator);

    for (map.keys(), map.values()) |key, info| {
        if (try createJob(heapAllocator, map, key, info)) |newJob| {
            try jobList.append(heapAllocator, newJob);
        }
    }

    // 13 tensors, the 3 satellites of a are skipped, we expect 10 jobs
    try std.testing.expectEqual(10, jobList.items.len);

    // null means skipped
    const Case = struct { key: []const u8, want: ?std.meta.Tag(Action) };
    const cases = [_]Case{
        // a: valid group, the weight carries the job, the others are skipped
        .{ .key = "a.weight", .want = .dequantize },
        .{ .key = "a.weight_scale", .want = null },
        .{ .key = "a.weight_scale_2", .want = null },
        .{ .key = "a.input_scale", .want = null },
        // b: no weight_scale_2, the group is not valid, so we keep everything (copied)
        .{ .key = "b.weight", .want = .copy },
        .{ .key = "b.weight_scale", .want = .copy },
        // c: scale shape [2, 2] does not match weight [2, 8] (needs [2, 1]), not valid (copied)
        .{ .key = "c.weight", .want = .copy },
        .{ .key = "c.weight_scale", .want = .copy },
        .{ .key = "c.weight_scale_2", .want = .copy },
        // d: BF16 scale, the group is not valid (copied)
        .{ .key = "d.weight", .want = .copy },
        .{ .key = "d.weight_scale", .want = .copy },
        .{ .key = "d.weight_scale_2", .want = .copy },
        // a plain tensor (copied)
        .{ .key = "norm.weight", .want = .copy },
    };

    for (cases) |case| {
        const job = try createJob(heapAllocator, map, case.key, map.get(case.key).?);
        const got: ?std.meta.Tag(Action) = if (job) |j| std.meta.activeTag(j.action) else null;
        std.testing.expectEqual(case.want, got) catch |err| {
            std.debug.print("wrong action for {s}\n", .{case.key});
            return err;
        };
    }

    const aJob = (try createJob(heapAllocator, map, "a.weight", map.get("a.weight").?)).?;
    try std.testing.expectEqual([2]u64{ 50, 114 }, aJob.action.dequantize.weight.data_offsets);
    try std.testing.expectEqual([2]u64{ 36, 44 }, aJob.action.dequantize.scale.data_offsets);
    try std.testing.expectEqual([2]u64{ 4, 8 }, aJob.action.dequantize.scale2.data_offsets);
}

test "make the jobs" {
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

    var parsedJson = try read.parseHeader(heapAllocator, jsonStr);
    defer parsedJson.deinit();
    _ = parsedJson.value.map.orderedRemove("__metadata__");

    const jobs = try makeJobs(heapAllocator, &parsedJson.value.map);
    defer heapAllocator.free(jobs);

    // Skips dropped, sorted by data_offset
    const want = [_][]const u8{
        "c.weight_scale_2",
        "d.weight_scale_2",
        "d.weight_scale",
        "norm.weight",
        "b.weight_scale",
        "c.weight_scale",
        "a.weight",
        "b.weight",
        "c.weight",
        "d.weight",
    };
    try std.testing.expectEqual(want.len, jobs.len);
    for (want, jobs) |name, job| {
        std.testing.expectEqualStrings(name, job.name) catch |err| {
            std.debug.print("wrong order at {s}\n", .{job.name});
            return err;
        };
    }
    try std.testing.expect(jobs[6].action == .dequantize);
}

// Something that comes along NVFP4 (scale, scale_2, input_scale) of a valid NVFP4 group gets no job, the weight's job uses it
pub fn isSkipped(allocator: std.mem.Allocator, map: *const std.StringArrayHashMapUnmanaged(read.TensorInfo), key: []const u8) !bool {
    return for ([_][]const u8{ ".weight_scale_2", ".weight_scale", ".input_scale" }) |suffix| {
        if (!std.mem.endsWith(u8, key, suffix)) continue;
        const weightName = try std.fmt.allocPrint(allocator, "{s}.weight", .{key[0 .. key.len - suffix.len]});
        defer allocator.free(weightName);
        break try findNvfp4(allocator, map, weightName) != null;
    } else false;
}

test "find skippable element" {
    // jsonStr generated by claude
    const jsonStr =
        \\{
        \\  "a.weight_scale_2": { "dtype": "F32",     "shape": [],      "data_offsets": [0, 4] },
        \\  "a.input_scale":    { "dtype": "F32",     "shape": [],      "data_offsets": [4, 8] },
        \\  "a.weight_scale":   { "dtype": "F8_E4M3", "shape": [2, 2],  "data_offsets": [8, 12] },
        \\  "a.weight":         { "dtype": "U8",      "shape": [2, 16], "data_offsets": [12, 44] },
        \\  "a.bias":           { "dtype": "BF16",    "shape": [2],     "data_offsets": [44, 48] },
        \\  "d.weight_scale_2": { "dtype": "F32",     "shape": [],      "data_offsets": [48, 52] },
        \\  "d.weight_scale":   { "dtype": "BF16",    "shape": [2, 1],  "data_offsets": [52, 56] },
        \\  "d.weight":         { "dtype": "U8",      "shape": [2, 8],  "data_offsets": [56, 72] },
        \\  "e.weight_scale_2": { "dtype": "F32",     "shape": [],      "data_offsets": [72, 76] },
        \\  "norm.weight":      { "dtype": "BF16",    "shape": [8],     "data_offsets": [76, 92] }
        \\}
    ;
    const heapAllocator = std.testing.allocator;
    const parsedJson = try read.parseHeader(heapAllocator, jsonStr);
    defer parsedJson.deinit();
    const map = &parsedJson.value.map;

    const Case = struct { key: []const u8, want: bool };
    const cases = [_]Case{
        // a: valid group, then the other should be skipped
        .{ .key = "a.weight_scale", .want = true },
        .{ .key = "a.weight_scale_2", .want = true },
        .{ .key = "a.input_scale", .want = true },
        // the weight itself carries the job, not skipped
        .{ .key = "a.weight", .want = false },
        // not part of a job, not skipped (copied)
        .{ .key = "a.bias", .want = false },
        // d: BF16 scale, the group is not valid, so we keep everything (copied)
        .{ .key = "d.weight_scale", .want = false },
        .{ .key = "d.weight_scale_2", .want = false },
        // e: a scale with no weight, nobody uses it (copied)
        .{ .key = "e.weight_scale_2", .want = false },
        // a plain tensor (copied)
        .{ .key = "norm.weight", .want = false },
    };

    for (cases) |case| {
        std.testing.expectEqual(case.want, try isSkipped(heapAllocator, map, case.key)) catch |err| {
            std.debug.print("wrong answer for {s}\n", .{case.key});
            return err;
        };
    }
}

test "slice string" {
    const key = "random_name.weights";
    const suffixId = std.mem.findScalarLast(u8, key, '.') orelse return;
    const out = key[0..suffixId];
    try std.testing.expectEqualStrings("random_name", out);
}

// Function to find the data linked with a set of weights for NVFP4
// If it returns null, it means its not NVFP4 and we either copy it or skip it
pub fn findNvfp4(allocator: std.mem.Allocator, map: *const std.StringArrayHashMapUnmanaged(read.TensorInfo), name: []const u8) !?Nvfp4Tensor {
    // If there's no weight, we return null
    if (!std.mem.endsWith(u8, name, ".weight")) return null;
    const weight = map.get(name) orelse return null;

    // Else there's weight, and we try to get the other field
    const dot = std.mem.findScalarLast(u8, name, '.') orelse return null;
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
    const parsedJson = try read.parseHeader(heapAllocator, jsonStr);
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

// Test the sorting function by job ofseets
test "sort jobs by input offset" {
    var jobs = [_]Job{
        .{ .name = "third.weights", .action = .{ .copy = .{ .data_offsets = .{ 130, 146 } } } },
        .{
            .name = "second.weights",
            .action = .{
                .dequantize = .{
                    .weight = .{ .data_offsets = .{ 50, 114 } },
                    .scale = .{ .data_offsets = .{ 200, 208 } }, // after everything: keying on it puts the job last
                    .scale2 = .{ .data_offsets = .{ 0, 4 } }, // before everything: keying on it puts the job first
                },
            },
        },
        .{ .name = "first.weights", .action = .{ .copy = .{ .data_offsets = .{ 10, 50 } } } },
    };

    std.mem.sort(Job, &jobs, {}, Job.byInputOffset);

    // Copies and dequantize interleaved by where they start reading, the dequantize keyed on its weight
    try std.testing.expectEqualStrings("first.weights", jobs[0].name);
    try std.testing.expectEqualStrings("second.weights", jobs[1].name);
    try std.testing.expectEqualStrings("third.weights", jobs[2].name);
}
