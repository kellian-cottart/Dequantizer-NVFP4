const std = @import("std");
const Io = std.Io;
const Dequantizer = @import("../dequantizer.zig").Dequantizer;

// One step per call: a piece of the header, or one mini-job of the current job
//fn stream();
