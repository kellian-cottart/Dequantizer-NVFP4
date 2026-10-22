const std = @import("std");
const math = std.math;

const e2m1Cases = @import("cases.zig").e2m1Cases;
const e4m3Cases = @import("cases.zig").e4m3Cases;
const nvfp4Cases = @import("cases.zig").nvfp4Cases;

// Function E2M1 SIMD, same but using SIMD instructions instead
// pub fn e2m1ToF32Simd(bits: @Vector(16, u32))
