const std = @import("std");
const Io = std.Io;
const read = @import("read.zig");
const Job = @import("dispatch/jobs.zig").Job;
const makeJobs = @import("dispatch/jobs.zig").makeJobs;
// const buildHeader = @import("dispatch/header.zig").buildHeader;
// const stream = @import("dispatch/stream.zig").stream;

// https://www.openmymind.net/Im-Too-Dumb-For-Zigs-New-IO-Interface/
// "I'm too dumb for Zig's new IO interface" me too bro, me too
pub const Dequantizer = struct {
    // The reader to stream the data
    interface: Io.Reader,

    // Where to read from
    allocator: std.mem.Allocator,
    io: Io,
    file: Io.File,

    // What the file contains: the job names point into it, so it lives as long as the jobs
    structure: read.Structure,

    // The plan, built once in init
    jobs: []Job,
    // the header to give back with the right sectors
    outputHeader: []u8, // 8-byte length + json (tensors + metadata), padded to 8,

    // Progress of the stream
    phase: enum { header, jobs } = .header,
    jobIndex: usize = 0,
    // .header: header bytes given out; .jobs: start of the next mini-job in the tensor
    pos: u64 = 0,
    chunkSize: usize, // packed bytes per mini-job, multiple of 8

    // // Read the structure, make the jobs, build the header
    // pub fn init(allocator: std.mem.Allocator, io: Io, file: Io.File, chunkSize: usize) !Dequantizer {
    //     const structure = try read.getStructure(allocator, io, file);
    //     errdefer structure.deinit();

    //     const jobs = try makeJobs(allocator, &structure.map.value.map);
    //     errdefer allocator.free(jobs);

    //     //const outputHeader = try buildHeader(allocator, jobs, structure.metadata.value);
    //     //errdefer allocator.free(outputHeader);

    //     // One mini-job decodes chunkSize packed bytes into chunkSize * 8 bytes of f32
    //     const buffer = try allocator.alloc(u8, chunkSize * 8);

    //     return .{
    //         .interface = .{ .vtable = &.{ .stream = stream }, .buffer = buffer, .seek = 0, .end = 0 },
    //         .allocator = allocator,
    //         .io = io,
    //         .file = file,
    //         .structure = structure,
    //         .jobs = jobs,
    //         .outputHeader = outputHeader,
    //         .chunkSize = chunkSize,
    //     };
    // }

    pub fn deinit(self: *Dequantizer) void {
        self.allocator.free(self.interface.buffer);
        self.allocator.free(self.outputHeader);
        self.allocator.free(self.jobs);
        self.structure.deinit();
    }
};

// Pull in the tests of the dispatch folder
test {
    _ = @import("dispatch/context.zig");
    _ = @import("dispatch/jobs.zig");
    _ = @import("dispatch/header.zig");
    _ = @import("dispatch/stream.zig");
}
