const std = @import("std");
const Io = std.Io;
const math = std.math;
const read = @import("Dequantizer").read;

pub const Action = union(enum) {
    copy, // Copy the inter
    dequantize, // Decode for NVFP4 values
    skip, // Skip the scales
};

pub const Job = struct {
    name: []const u8, // Which tensor will we tend to
    info: read.TensorInfo,
    action: Action,
    new_info: read.TensorInfo,
};

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

        return .{
            .interface = .{ .vtable = &.{ .stream = stream }, .buffer = buffer, .seek = 0, .end = 0 },
            .allocator = allocator,
            .io = io,
            .file = file,
            .structure = structure,
            .jobs = jobs,
            .outputHeader = outputHeader,
            .chunkSize = chunkSize,
        };
    }

    pub fn deinit(self: *Dequantizer) void {
        self.allocator.free(self.interface.buffer);
        self.allocator.free(self.outputHeader);
        self.allocator.free(self.jobs);
        self.structure.deinit();
    }

    // Tag every tensor (copy / dequantize / skip), drop the skips, sort by position
    // fn makeJobs(allocator: std.mem.Allocator, map: *const std.StringArrayHashMapUnmanaged(read.TensorInfo)) ![]Job {
    //     var jobList = std.ArrayList(Job).init(allocator);
    //     defer jobList.deinit();
    //     // get the jobs from map
    //     return jobList;
    // }

    // New header from the jobs: new dtypes and shapes, new offsets, plus the metadata
    //fn buildHeader();

    // One step per call: a piece of the header, or one mini-job of the current job
    //fn stream();
};

test "tag jobs" {}
