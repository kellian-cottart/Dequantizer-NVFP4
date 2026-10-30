const std = @import("std");
const Io = std.Io;
const read = @import("read.zig");
const Job = @import("dispatch/jobs.zig").Job;
const makeJobs = @import("dispatch/jobs.zig").makeJobs;
const buildHeader = @import("dispatch/header.zig").buildHeader;
const simd = @import("backend/simd/multiply.zig");

// Pull in the tests of the dispatch folder
test {
    _ = @import("dispatch/jobs.zig");
    _ = @import("dispatch/header.zig");
    _ = @import("backend/simd/e2m1.zig");
    _ = @import("backend/simd/e4m3.zig");
    _ = @import("backend/simd/multiply.zig");
}
pub const Dequantizer = struct {
    // The reader to stream the data
    // https://github.com/ziglang/zig/blob/master/lib/std/Io/Reader.zig
    // \lib\std\Io\Reader.zig
    // Given the doc, it has four fields: vtable, buffer, seek and end.
    // Vtable is a pointer to a struct
    // buffer is a pointer to a u8 array which is the number of bytes to be consumed from buffer
    // seek is a usize, the current position in the buffer
    // the stream is written in the vtable and must be

    // interface: to stream the data, it is a reader with a vtable that has a stream function
    // allocator: for memory,
    // io: for reading,
    // file: the file to read from
    // structure: to know which blocks to read
    // jobs: an array of jobs to perform, jobIndex: the current job, chunkSize: the size of the mini-job
    // err: an optional error that can be set if a read fails
    // scale2: we add it in case of a dequantize job for the second scale, stays across a job
    // default to 1.0 (neutral in the multiplication)
    // scaleBuffer: the E4M3 block scales of the current mini-job, one byte per block of 16 values
    // weightBuffer: packed E2M1 weights of the current mini-job, two values per byte (u8 instead of u4)
    // pos: serves to indicate the starting point of the mini-job
    // numWorkers: how many workers decode a mini-job in parallel, at most the cpu count
    // blocksPerPlate: how many blocks a worker takes per serving (256 measured best, check claude benchmark)
    // minBlocksForWorkers: below this many blocks in a mini-job, decode on the calling thread (8192 measured)
    // outputHeader: the header of the output file (8 bytes of length + json), streamed before the first job
    // headerPos: how much of the header was already streamed

    interface: Io.Reader,
    allocator: std.mem.Allocator,
    io: Io,
    file: Io.File,
    structure: read.Structure,
    scale2: f32 = 1.0,
    scaleBuffer: []u8,
    weightBuffer: []u8,
    jobs: []Job,
    jobIndex: usize = 0,
    chunkSize: usize,
    numWorkers: usize,
    blocksPerPlate: usize,
    minBlocksForWorkers: usize,
    outputHeader: []u8,
    headerPos: usize = 0,
    pos: u64 = 0,
    err: ?Io.File.ReadPositionalError = null,

    // Read the structure, make the jobs, build the header
    pub fn init(allocator: std.mem.Allocator, io: Io, file: Io.File, chunkSize: usize, numWorkers: usize, blocksPerPlate: usize, minBlocksForWorkers: usize) !Dequantizer {
        // chunkSize is always a multiple of 8
        if (chunkSize == 0 or chunkSize % 8 != 0) return error.InvalidChunkSize;
        // at least one worker, and never more than the cpus (hardware threads), 1 if the count fails
        if (numWorkers == 0) return error.InvalidNumWorkers;
        // a plate of 0 blocks would never advance
        if (blocksPerPlate == 0) return error.InvalidBlocksPerPlate;
        const cpuCount = std.Thread.getCpuCount() catch 1;

        const structure = try read.getStructure(allocator, io, file);
        errdefer structure.deinit();

        const jobs = try makeJobs(allocator, &structure.map.value.map);
        errdefer allocator.free(jobs);

        const outputHeader = try buildHeader(allocator, jobs, structure.metadata.value);
        errdefer allocator.free(outputHeader);

        // One mini-job decodes chunkSize packed bytes into chunkSize * 8 bytes of f32
        const buffer = try allocator.alloc(u8, chunkSize * 8);
        errdefer allocator.free(buffer);

        // buffer for the scales
        const scaleBuffer = try allocator.alloc(u8, chunkSize / 8);
        errdefer allocator.free(scaleBuffer);

        // buffer for the weights
        const weightBuffer = try allocator.alloc(u8, chunkSize);
        errdefer allocator.free(weightBuffer);

        // The Vtable is a struct with a field stream,
        //which is a function pointer to the stream function

        return .{
            .interface = .{ .vtable = &.{ .stream = stream }, .buffer = buffer, .seek = 0, .end = 0 },
            .allocator = allocator,
            .io = io,
            .file = file,
            .scaleBuffer = scaleBuffer,
            .weightBuffer = weightBuffer,
            .structure = structure,
            .jobs = jobs,
            .outputHeader = outputHeader,
            .chunkSize = chunkSize,
            .numWorkers = @min(numWorkers, cpuCount),
            .blocksPerPlate = blocksPerPlate,
            .minBlocksForWorkers = minBlocksForWorkers,
        };
    }

    pub fn deinit(self: *Dequantizer) void {
        self.allocator.free(self.interface.buffer);
        self.allocator.free(self.scaleBuffer);
        self.allocator.free(self.weightBuffer);
        self.allocator.free(self.outputHeader);
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
        // w and limit are only used for the header, the jobs go through r.buffer

        // Retrieve the parent struct from the pointer to the reader
        const self: *Dequantizer = @fieldParentPtr("interface", r);

        // when we have consumed all the bytes in the buffer
        if (r.seek == r.end) {
            r.seek = 0;
            r.end = 0;
        }

        // when streaming starts, the header of the output file goes straight to the writer, before the first job
        if (self.headerPos < self.outputHeader.len) {
            const written = try w.write(limit.slice(self.outputHeader[self.headerPos..]));
            self.headerPos += written;
            return written;
        }

        // if the jobs are finished, then we don't have to stream anymore
        if (self.jobIndex >= self.jobs.len) return error.EndOfStream;

        // Retrieve the job to be done
        const currentJob = self.jobs[self.jobIndex];

        // do something depending on the job type
        switch (currentJob.action) {
            .copy => |info| {
                // get the length of the copy job
                const copyLen = info.data_offsets[1] - info.data_offsets[0];
                // Take the space remaining in the buffer (could have also used limit)
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
                    const nbBytes = self.file.readPositionalAll(self.io, &scale2Bytes, scale2Start) catch |err| {
                        self.err = err;
                        return error.ReadFailed;
                    };
                    if (nbBytes != 4) return error.ReadFailed;
                    self.scale2 = @bitCast(std.mem.readInt(u32, &scale2Bytes, .little));
                }

                const buffer = r.buffer[r.end..];
                const copyLen = nvfp4Info.weight.data_offsets[1] - nvfp4Info.weight.data_offsets[0];
                // Since here, we're going to push f32 instead of u8,
                // (u4 x 2 due to having the lower nibble and upper nibble)
                // we need to divide the size of a mini-job by 8 so that everything fitst in
                const miniJobSize: usize = @intCast(@min(self.chunkSize, copyLen - self.pos, buffer.len / 8));

                // Set-up the scales for the mini-job
                const scaleBufferSlice = self.scaleBuffer[0 .. miniJobSize / 8];
                const scaleStart = self.structure.dataStart + nvfp4Info.scale.data_offsets[0] + self.pos / 8;
                // we read to that position in the bytes array
                const nbBytesScales = self.file.readPositionalAll(self.io, scaleBufferSlice, scaleStart) catch |err| {
                    self.err = err;
                    return error.ReadFailed;
                };
                if (nbBytesScales != scaleBufferSlice.len) return error.ReadFailed;

                // Set-up the weights for the minijob

                const weights = self.weightBuffer[0..miniJobSize];
                const weightStart = self.structure.dataStart + nvfp4Info.weight.data_offsets[0] + self.pos;
                const nbBytesWeights = self.file.readPositionalAll(self.io, weights, weightStart) catch |err| {
                    self.err = err;
                    return error.ReadFailed;
                };
                if (nbBytesWeights != miniJobSize) return error.ReadFailed;

                // now we can compute the mini-jobs sequentially and write them directly into the writing buffer
                simd.bufferToWorker(self.io, weights, scaleBufferSlice, self.scale2, buffer, self.numWorkers, self.blocksPerPlate, self.minBlocksForWorkers) catch |err| {
                    self.err = err;
                    return error.ReadFailed;
                };
                // std.debug.print("buffer {any} \n", .{self.weightBuffer[0..miniJobSize]});
                // std.debug.print("scale buffer {any} \n", .{scaleBufferSlice});

                // Update the position
                self.pos += miniJobSize;
                // we send out float32 instead so we need 8 times more space
                r.end += miniJobSize * 8;
                if (self.pos == copyLen) {
                    self.jobIndex += 1;
                    self.pos = 0;
                }
            },
        }
        return 0;
    }
};

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
    // generated by claude
    const json = "{\"x\":{\"dtype\":\"BF16\",\"shape\":[8],\"data_offsets\":[0,16]}}";
    const body = [16]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    // Create a temp dir to hold the fake file
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // Write the fake file
    const file = try writeFakeFile(io, tmp.dir, json, &body);
    defer file.close(io);

    // Using the dequantizer with 8 chunksize and 1 gpu
    var dequantizer = try Dequantizer.init(allocator, io, file, 8, 1, 256, 8192);
    defer dequantizer.deinit();

    // Exactly one copy job of 16 bytes
    try std.testing.expectEqual(1, dequantizer.jobs.len);
    try std.testing.expect(dequantizer.jobs[0].action == .copy);
    const info = dequantizer.jobs[0].action.copy;
    try std.testing.expectEqual(16, info.data_offsets[1] - info.data_offsets[0]);

    // The 16 bytes come out unchanged
    // room for the header (built in init, its content is tested in dispatch/header.zig) and the 16 bytes
    var output: [512]u8 = undefined;
    var writer: Io.Writer = .fixed(&output);
    const nbBytes = try dequantizer.interface.streamRemaining(&writer);
    const headerLength = dequantizer.outputHeader.len;

    // expect the header then the job
    try std.testing.expectEqual(headerLength + 16, nbBytes);
    try std.testing.expectEqualSlices(u8, dequantizer.outputHeader, writer.buffered()[0..headerLength]);
    // expect the data to match the input
    try std.testing.expectEqualSlices(u8, &body, writer.buffered()[headerLength..]);
}

test "dequantize stream" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // this is the data from the test file
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
    // generated by claude
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
    var dequantizer = try Dequantizer.init(allocator, io, file, 8, 1, 256, 8192);
    defer dequantizer.deinit();

    // the header, then 16 bytes of the norm copy, then 32 packed bytes x 2 values x 4 bytes of f32 = 256
    var output: [1024]u8 = undefined;
    var writer: Io.Writer = .fixed(&output);
    const nbBytes = try dequantizer.interface.streamRemaining(&writer);
    const headerLength = dequantizer.outputHeader.len;
    try std.testing.expectEqual(headerLength + 272, nbBytes);
    try std.testing.expectEqualSlices(u8, dequantizer.outputHeader, writer.buffered()[0..headerLength]);
    // the data starts right after the header
    const data = writer.buffered()[headerLength..];

    // the norm weight is copied unchanged
    try std.testing.expectEqualSlices(u8, body[8..24], data[0..16]);

    // the dequantized weight: e2m1 x e4m3 x scale2 for each nibble, low nibble first
    // scale2 = 3.7202382e-05 (bytes 193, 9, 28, 56), one e4m3 scale per 8 bytes
    // computed exactly then rounded once to f32 (generated by claude)
    const answers = [64]f32{
        // block 0: scale 0x38 = 1.0, bytes 16, 50, 84, 118, 152, 186, 220, 254
        0.0,           0.000018601191,    0.000037202382,   0.000055803575,    0.000074404765,   0.00011160715,     0.00014880953,    0.0002232143,
        -0.0,          -0.000018601191,   -0.000037202382,  -0.000055803575,   -0.000074404765,  -0.00011160715,    -0.00014880953,   -0.0002232143,
        // block 1: scale 0x3F = 1.875, bytes 239, 205, 171, 137, 103, 69, 35, 1
        -0.0004185268, -0.00027901787,    -0.0002092634,    -0.00013950893,    -0.0001046317,    -0.00006975447,    -0.000034877234,  -0.0,
        0.0004185268,  0.00027901787,     0.0002092634,     0.00013950893,     0.0001046317,     0.00006975447,     0.000034877234,   0.0,
        // block 2: scale 0x01 = 2^-9 (subnormal), bytes 16, 50, 84, 118, 152, 186, 220, 254
        0.0,           0.00000003633045,  0.0000000726609,  0.00000010899136,  0.0000001453218,  0.00000021798272,  0.0000002906436,  0.00000043596543,
        -0.0,          -0.00000003633045, -0.0000000726609, -0.00000010899136, -0.0000001453218, -0.00000021798272, -0.0000002906436, -0.00000043596543,
        // block 3: scale 0x7E = 448.0, bytes 247 (0xF7: +6 then -6)
        0.1,           -0.1,              0.1,              -0.1,              0.1,              -0.1,              0.1,              -0.1,
        0.1,           -0.1,              0.1,              -0.1,              0.1,              -0.1,              0.1,              -0.1,
    };

    var index: usize = 0;
    while (index < answers.len) : (index += 1) {
        // bitcasting to f32 on each group of four bits
        const current: f32 = @bitCast(std.mem.readInt(u32, data[16 + 4 * index ..][0..4], .little));
        std.testing.expectApproxEqRel(answers[index], current, 1e-6) catch |err| {
            return err;
        };
    }
}
