const std = @import("std");
const Io = std.Io;
const read = @import("read.zig");
const Job = @import("dispatch/jobs.zig").Job;
const makeJobs = @import("dispatch/jobs.zig").makeJobs;

pub const Dequantizer = struct {
    // The reader to stream the data
    // https://github.com/ziglang/zig/blob/master/lib/std/Io/Reader.zig
    // \lib\std\Io\Reader.zig
    // Given the doc, it has four fields: vtable, buffer, seek and end.
    // Vtable is a pointer to a struct
    // buffer is a pointer to a u8 array which is the number of bytes to be consumed from buffer
    // seek is a usize, the current position in the buffer
    // the stream is written in the vtable and must be

    // Interface to stream the data, it is a reader with a vtable that has a stream function
    // Allocator for memory, io for reading, file for reading
    // stucture to know which blocks to read
    // jobs is an array of jobs to perform, jobIndex is the current job, chunkSize is the size of the mini-job
    // err is an optional error that can be set if a read fails
    // we add scale2 in case of a dequantize job
    // this conserves the information accross mini-jobs
    // 1.0 is neutral in the multiplication if its not set
    // pos serves to indicate the starting point of the mini-job

    interface: Io.Reader,
    allocator: std.mem.Allocator,
    io: Io,
    file: Io.File,
    structure: read.Structure,
    scale2: f32 = 1.0,
    jobs: []Job,
    jobIndex: usize = 0,
    chunkSize: usize,
    pos: u64 = 0,
    err: ?Io.File.ReadPositionalError = null,

    // Read the structure, make the jobs, build the header
    pub fn init(allocator: std.mem.Allocator, io: Io, file: Io.File, chunkSize: usize) !Dequantizer {
        const structure = try read.getStructure(allocator, io, file);
        errdefer structure.deinit();

        const jobs = try makeJobs(allocator, &structure.map.value.map);
        errdefer allocator.free(jobs);

        //const outputHeader = try buildHeader(allocator, jobs, structure.metadata.value);
        //errdefer allocator.free(outputHeader);

        // One mini-job decodes chunkSize packed bytes into chunkSize * 8 bytes of f32
        const buffer = try allocator.alloc(u8, chunkSize * 8);
        errdefer allocator.free(buffer);

        // The Vtable is a struct with a field stream,
        //which is a function pointer to the stream function

        return .{
            .interface = .{ .vtable = &.{ .stream = stream }, .buffer = buffer, .seek = 0, .end = 0 },
            .allocator = allocator,
            .io = io,
            .file = file,
            .structure = structure,
            .jobs = jobs,
            // .outputHeader = outputHeader, // outputheader will not be computed for now
            .chunkSize = chunkSize,
        };
    }

    pub fn deinit(self: *Dequantizer) void {
        self.allocator.free(self.interface.buffer);
        // self.allocator.free(self.outputHeader);
        self.allocator.free(self.jobs);
        self.structure.deinit();
    }

    // https://github.com/ziglang/zig/blob/master/lib/std/Io/Reader.zig
    // https://cookbook.ziglang.cc/io/
    // I need to make the stream able to handle copy/dequantize jobs
    fn stream(r: *Io.Reader, w: *Io.Writer, limit: Io.Limit) Io.Reader.StreamError!usize {

        // stream is called within streamRemaining
        // what it does is that it calls the stream funct that will drop a most one mini-job in the r.buffer
        // (see reader interface)
        _ = w;
        _ = limit;

        // Retrieve the parent struct from the pointer to the reader
        const self: *Dequantizer = @fieldParentPtr("interface", r);

        // if the jobs are finished, then we don't have to stream anymore
        if (self.jobIndex >= self.jobs.len) return error.EndOfStream;

        // Retrieve the job to be done
        const currentJob = self.jobs[self.jobIndex];

        // when we have consumed all the bytes in the buffer
        if (r.seek == r.end) {
            r.seek = 0;
            r.end = 0;
        }

        // do something depending on the job type
        switch (currentJob.action) {
            .copy => |info| {
                // get the length of the copy job
                const copyLen = info.data_offsets[1] - info.data_offsets[0];
                // Free the space already taken in the buffer
                const buffer = r.buffer[r.end..];

                // Get the real length of the mini-job, which is the minimum between the chunkSize and the bytes left to copy.
                // we also make the buffer inbounds
                const miniJobSize: usize = @intCast(@min(self.chunkSize, copyLen - self.pos, buffer.len));

                // We're still on the same job, hence we can gather our datastart which is 8 + len header
                // then add the position of the beginning of the current job, then the beginning of the current mini-job
                const startingPoint = self.structure.dataStart + info.data_offsets[0] + self.pos;

                // Read the mini-job from the file into the buffer
                // https://www.mtsoukalos.eu/2026/04/file-seeking-in-zig/
                // https://www.ziglang.in/learn/files/random-access/
                // params: io, buffer, startingPoint
                const output = self.file.readPositionalAll(self.io, buffer[0..miniJobSize], startingPoint) catch |err| {
                    self.err = err;
                    return error.ReadFailed;
                };

                if (output != miniJobSize) {
                    return error.ReadFailed;
                }
                // Update the position
                self.pos += miniJobSize;
                r.end += miniJobSize;

                // if we have done all the minijobs, then we can move to the next job
                if (self.pos == copyLen) {
                    self.jobIndex += 1;
                    self.pos = 0;
                }
            },
            .dequantize => |nvfp4Info| {
                // reminded: nvfp4info has three field: scale, scale2 and weight that are TensorInfo structs
                // cf: dispatch/jobs
                // here we have a dequantize job, hence we need to read scales and weights
                // first, we retrieve the scale 2 at the beginning of the job
                if (self.pos == 0) {
                    // f32 = 4 x u8
                    var scale2Bytes: [4]u8 = undefined;
                    const scale2Start = self.structure.dataStart + nvfp4Info.scale2.data_offsets[0];
                    // we read to that position in the bytes array
                    self.file.readPositionalAll(self.io, &scale2Bytes, scale2Start) catch |err| {
                        self.err = err;
                        return error.ReadFailed;
                    };
                    self.scale2 = @bitCast(std.mem.readInt(u32, &scale2Bytes, .little));
                }

                // for the bytes of the scale they're

                const buffer = r.buffer[r.end..];

                const copyLen = nvfp4Info.weight.data_offsets[1] - nvfp4Info.weight.data_offsets[0];

                const miniJobSize: usize = @intCast(@min(self.chunkSize, copyLen - self.pos, buffer.len));
                const startingPoint = self.structure.dataStart + nvfp4Info.weight.data_offsets[0] + self.pos;

                const output = self.file.readPositionalAll(self.io, buffer[0..miniJobSize], startingPoint) catch |err| {
                    self.err = err;
                    return error.ReadFailed;
                };

                if (output != miniJobSize) {
                    return error.ReadFailed;
                }
                // Update the position
                self.pos += miniJobSize;
                r.end += miniJobSize;
                if (self.pos == copyLen) {
                    self.jobIndex += 1;
                    self.pos = 0;
                }
            },
        }
        return 0;
    }
};

// line 1840-1848 of the reader.zig file
//   fn stream(r: *Reader, w: *Writer, limit: Limit) StreamError!usize {
//         const this: *@This() = @alignCast(@fieldParentPtr("reader", r));
//         const data = limit.slice(try w.writableSliceGreedy(1));
//         var vec: [1][]u8 = .{data};
//         const n = try this.in.readVec(&vec);
//         this.hasher.update(data[0..n]);
//         w.advance(n);
//         return n;
//     }

// Pull in the tests of the dispatch folder
test {
    _ = @import("dispatch/jobs.zig");
    _ = @import("backend/simd.zig");
}

// Write a fake safetensors file: u64 little-endian header length, the json header, then the body
fn writeFakeFile(io: Io, dir: Io.Dir, json: []const u8, body: []const u8) !Io.File {
    const file = try dir.createFile(io, "fake.safetensors", .{ .read = true });
    errdefer file.close(io);
    const headerLen = std.mem.toBytes(std.mem.nativeToLittle(u64, json.len));
    try file.writeStreamingAll(io, &headerLen);
    try file.writeStreamingAll(io, json);
    try file.writeStreamingAll(io, body);
    return file;
}

test "copy stream" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // One BF16 tensor of 8 values = 16 bytes: a single copy job
    const json = "{\"x\":{\"dtype\":\"BF16\",\"shape\":[8],\"data_offsets\":[0,16]}}";
    const body = [16]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    // Create a temp dir to hold the fake file
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // Write the fake file
    const file = try writeFakeFile(io, tmp.dir, json, &body);
    defer file.close(io);

    // chunkSize 8: the 16-byte job takes 2 mini-jobs, so pos and the job rollover get exercised
    var dequantizer = try Dequantizer.init(allocator, io, file, 8);
    defer dequantizer.deinit();

    // Exactly one copy job of 16 bytes
    try std.testing.expectEqual(1, dequantizer.jobs.len);
    try std.testing.expect(dequantizer.jobs[0].action == .copy);
    const info = dequantizer.jobs[0].action.copy;
    try std.testing.expectEqual(16, info.data_offsets[1] - info.data_offsets[0]);

    // The 16 bytes come out unchanged
    var output: [16]u8 = undefined;
    var writer: Io.Writer = .fixed(&output);
    const nbBytes = try dequantizer.interface.streamRemaining(&writer);

    // expect the job to be done and the output to match the input
    try std.testing.expectEqual(16, nbBytes);
    // expect the output to match the input
    try std.testing.expectEqualSlices(u8, &body, writer.buffered());
}

test "dequantize stream" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // this is the data from the t
    const json =
        \\{
        \\  "__metadata__":        { "format": "pt" },
        \\  "tiny.input_scale":    { "dtype": "F32",     "shape": [],      "data_offsets": [0, 4] },
        \\  "tiny.weight_scale_2": { "dtype": "F32",     "shape": [],      "data_offsets": [4, 8] },
        \\  "tiny.norm.weight":    { "dtype": "BF16",    "shape": [8],     "data_offsets": [8, 24] },
        \\  "tiny.weight_scale":   { "dtype": "F8_E4M3", "shape": [2, 2],  "data_offsets": [24, 28] },
        \\  "tiny.weight":         { "dtype": "U8",      "shape": [2, 16], "data_offsets": [28, 60] }
        \\}
    ;

    // Body of testdata/tiny_nvfp4.safetensors (60 bytes)
    const body = [60]u8{
        // tiny.input_scale [0, 4]: F32 0.5
        0,   0,   0,   63,
        // tiny.weight_scale_2 [4, 8]: F32
        193, 9,   28,  56,
        // tiny.norm.weight [8, 24]: 8 x BF16
        128, 191, 55,  191,
        219, 190, 18,  190,
        18,  62,  219, 62,
        55,  63,  128, 63,
        // tiny.weight_scale [24, 28]: 2 x 2 E4M3
        56,  63,  1,   126,
        // tiny.weight [28, 60]: 2 x 16 U8 (2 FP4 nibbles each)
        16,  50,  84,  118,
        152, 186, 220, 254,
        239, 205, 171, 137,
        103, 69,  35,  1,
        16,  50,  84,  118,
        152, 186, 220, 254,
        247, 247, 247, 247,
        247, 247, 247, 247,
    };

    // Create a temp dir to hold the fake file
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // Write the fake file
    const file = try writeFakeFile(io, tmp.dir, json, &body);
    defer file.close(io);

    // chunkSize 8: the 16-byte job takes 2 mini-jobs, so pos and the job rollover get exercised
    var dequantizer = try Dequantizer.init(allocator, io, file, 8);
    defer dequantizer.deinit();

    // Exactly one copy job of 16 bytes
    try std.testing.expectEqual(1, dequantizer.jobs.len);
    try std.testing.expect(dequantizer.jobs[0].action == .copy);
    const info = dequantizer.jobs[0].action.copy;
    try std.testing.expectEqual(16, info.data_offsets[1] - info.data_offsets[0]);

    // The 16 bytes come out unchanged
    var output: [16]u8 = undefined;
    var writer: Io.Writer = .fixed(&output);
    const nbBytes = try dequantizer.interface.streamRemaining(&writer);

    // expect the job to be done and the output to match the input
    try std.testing.expectEqual(16, nbBytes);
    // expect the output to match the input
    try std.testing.expectEqualSlices(u8, &body, writer.buffered());
}
