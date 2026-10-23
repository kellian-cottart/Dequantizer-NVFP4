const std = @import("std");
const Io = std.Io;
const read = @import("read.zig");
const Job = @import("dispatch/jobs.zig").Job;
const makeJobs = @import("dispatch/jobs.zig").makeJobs;

// const buildHeader = @import("dispatch/header.zig").buildHeader;

// https://www.openmymind.net/Im-Too-Dumb-For-Zigs-New-IO-Interface/
// "I'm too dumb for Zig's new IO interface" me too bro, me too
pub const Dequantizer = struct {
    // The reader to stream the data
    // https://github.com/ziglang/zig/blob/master/lib/std/Io/Reader.zig
    // \lib\std\Io\Reader.zig
    // Given the doc, it has four fields: vtable, buffer, seek and end.
    // Vtable is a pointer to a struct
    // buffer is a pointer to a u8 array which is the number of bytes to be consumed from buffer
    // seek is a usize, the current position in the buffer
    // the stream is written in the vtable and must be
    interface: Io.Reader,

    // Where to read from
    allocator: std.mem.Allocator,
    io: Io,
    file: Io.File,

    // What the file contains: the job names point into it, so it lives as long as the jobs
    structure: read.Structure,

    // The plan, built once in init
    jobs: []Job,

    // Progress of the stream
    jobIndex: usize = 0,

    // start of the next mini-job in the tensor
    pos: u64 = 0,

    // The number of packed bytes per mini-job, multiple of 8
    chunkSize: usize,

    // adding an error field if not readPositionalALl won't work due to not having the error propagated
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
                // here we have a dequantize job, hence we need to read scales and weights

                // read the scales
                const scale = try self.file.readPositionalAll(self.io, r.buffer[0..8], self.structure.dataStart + nvfp4Info.scale.data_offsets[0]) catch |err| {
                    self.err = err;
                    return error.ReadFailed;
                };

                const copyLen = nvfp4Info.weight.data_offsets[1] - nvfp4Info.weight.data_offsets[0];
                const buffer = r.buffer[r.end..];
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
