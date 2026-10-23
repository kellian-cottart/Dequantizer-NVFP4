const std = @import("std");
const math = std.math;

const e2m1Cases = @import("scalar.zig").e2m1Cases;
const e4m3Cases = @import("scalar.zig").e4m3Cases;
const nvfp4Cases = @import("scalar.zig").nvfp4Cases;

// SIMD, what happens in there?
// https://www.openmymind.net/SIMD-With-Zig/
// @Vector(length, type): Fixed-size vector of primitives (integers, floats, booleans or pointers)
// length must be known at compile time. arithmetics happen on a "lane"
// two 8 lanes vectors can allow for 8 sums done in one instruction instead of 8
// @splat(value): copies a value into every lane, will be useful for the 16 values of a NVFP4 block
// @select(type, mask, a, b): it's an if, will be useful when selecting the subnormal or not
// @reduce(op, v): collapses a vector into an operation giving one element, like in Hadoop

// NVFP4 blocks are 16 values sharing one E4M3 scales, so it makes sense to do 16 lanes for both methods
// My Mini-jobs then can be a multiple of 8

// For e2m1, it's pretty simple
// We actually know all cases, so we can just do a LUT

pub fn e2m1ToF32SimdSlow(bits: @Vector(16, u4)) @Vector(16, f32) {
    // Float cannot emerge from u4 natively,
    // we need first to widen to perform operations between types
    const V = @Vector(16, u32);
    const F = @Vector(16, f32);
    const wide: V = bits;

    // Same as before, we bit shift
    const sign = wide >> @splat(3);
    const exponent = (wide >> @splat(1)) & @as(V, @splat(0b11));
    const mantissa = wide & @as(V, @splat(1));

    // the mantissa goes to float
    const floatMantissa: @Vector(16, f32) = @floatFromInt(mantissa);

    // is it the normal or subnormal path? we convert it to float
    const isNormal: F = @floatFromInt(@intFromBool(exponent != @as(V, @splat(0))));
    // @exp2 only takes in power of two, and floats, hence we need to convert to
    const floatExponent: F = @floatFromInt(@max(exponent, @as(V, @splat(1))));
    // the exponent function is usually super slow and will be the first bottleneck, let's see if we can't bit shift that someway else or LUT
    const magnitude = @exp2(floatExponent - @as(F, @splat(1.0))) * (isNormal + floatMantissa * @as(F, @splat(0.5)));

    // The sign is the top bit of the float: 0 with the sign set is -0.0
    return @bitCast(@as(V, @bitCast(magnitude)) | (sign << @splat(31)));
}

// doing it with LUT is the quickest way to go about it
pub const e2m1LUT = [16]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 };

pub fn e2m1ToF32SimdLut(bits: @Vector(16, u4)) @Vector(16, f32) {
    var out: [16]f32 = undefined;
    // Each lane picks its value in the table, the u4 index can never go out of bounds
    inline for (0..16) |i| {
        out[i] = e2m1LUT[bits[i]];
    }
    return out;
}

// For computing the exponent, we can pose the maths
// (I would like to thank Theo Ballet for his expertise and taking time
// to explain to me like a four years old how it all works in the non-binary world)
// s·2^3 + e·2^1 + m·2^0 for seem
// s′·2^31 + E′·2^23 + M′ for f32
// Let's set an equality between the two
// (−1)^s′ = (−1)^s gives s′ = s
// 2^(E′−127) = 2^(e−1) gives E′ = e + 126
// M′/2^23 = m/2 gives M′ = m·2^22

pub fn e2m1ToF32Simd(bits: @Vector(16, u4)) @Vector(16, f32) {
    const V = @Vector(16, u32);

    // Widen to u32 lanes, so every field can move to its place in the f32
    const wide: V = bits;
    const sign = wide >> @splat(3);
    const exponent = (wide >> @splat(1)) & @as(V, @splat(0b11));
    const mantissa = wide & @as(V, @splat(1));

    // Normal: f32 exponent is e - 1 + 127, the mantissa bit becomes the top mantissa bit (bit 22)
    const normal = ((exponent + @as(V, @splat(126))) << @splat(23)) | (mantissa << @splat(22));
    // Subnormal: 0 or 0.5, and 0.5 is an f32 exponent of 126 with mantissa 0
    // mantissa is either 0 or 1, it gives either 0 or 1/2, which are needed for our computations
    const subnormal = mantissa * @as(V, @splat(126 << 23));
    const magnitude = @select(u32, exponent == @as(V, @splat(0)), subnormal, normal);

    // The sign is the top bit of the float: 0 with the sign set is -0.0
    return @bitCast(magnitude | (sign << @splat(31)));
}

test "block of 16 random values" {
    const blockCases = [_]struct { bits: @Vector(16, u4), want: @Vector(16, f32) }{.{
        .bits = .{ 0x7, 0x7, 0x0, 0xF, 0x8, 0x1, 0x9, 0x2, 0xA, 0x3, 0xB, 0x4, 0xC, 0x5, 0xD, 0x6 },
        .want = .{ 6.0, 6.0, 0.0, -6.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 4.0 },
    }};
    for (blockCases) |case| {
        std.testing.expectEqual(@as(@Vector(16, u32), @bitCast(case.want)), @as(@Vector(16, u32), @bitCast(e2m1ToF32Simd(case.bits)))) catch |err| {
            std.debug.print("failed on 0x{X}\n", .{case.bits});
            return err;
        };
    }
}

test "block of all 16 values" {
    const blockCases = [_]struct { bits: @Vector(16, u4), want: @Vector(16, f32) }{.{
        .bits = .{ 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF },
        .want = .{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 },
    }};

    for (blockCases) |case| {
        std.testing.expectEqual(@as(@Vector(16, u32), @bitCast(case.want)), @as(@Vector(16, u32), @bitCast(e2m1ToF32Simd(case.bits)))) catch |err| {
            std.debug.print("failed on 0x{X}\n", .{case.bits});
            return err;
        };
    }
}

// Timing operations were fully generated by claude using the log.md info i had set for the sake of knowing what i'm doing
// Time one implementation on the same blocks: best of 5 runs, in nanoseconds per code
fn timeE2m1(io: std.Io, comptime decode: fn (@Vector(16, u4)) @Vector(16, f32), blocks: []const [16]u8, out: []@Vector(16, f32)) f64 {
    const passes = 1000;
    var best: u64 = math.maxInt(u64);
    for (0..5) |_| {
        const start = std.Io.Timestamp.now(io, .awake);
        for (0..passes) |_| {
            // one code per byte, as the nibble split will give them, narrowed to u4 lanes
            for (blocks, out) |block, *o| o.* = decode(@truncate(@as(@Vector(16, u8), block)));
            // nobody reads out, so without this the compiler could skip the work
            std.mem.doNotOptimizeAway(out.ptr);
        }
        const ns: u64 = @intCast(start.untilNow(io, .awake).nanoseconds);
        best = @min(best, ns);
    }
    return @as(f64, @floatFromInt(best)) / @as(f64, @floatFromInt(passes * blocks.len * 16));
}

test "time the e2m1 implementations" {
    // Timings in Debug mean nothing, run with: zig test -OReleaseFast -mcpu=native src/backend/simd.zig
    if (@import("builtin").mode == .debug) return error.SkipZigTest;

    const io = std.testing.io;
    const heapAllocator = std.testing.allocator;

    // 1024 blocks of random codes, small enough to stay in cache: we time the decode, not the memory
    const blocks = try heapAllocator.alloc([16]u8, 1024);
    defer heapAllocator.free(blocks);
    const out = try heapAllocator.alloc(@Vector(16, f32), 1024);
    defer heapAllocator.free(out);

    var prng = std.Random.DefaultPrng.init(42);
    for (blocks) |*block| {
        inline for (0..16) |i| block.*[i] = prng.random().int(u4);
    }
    std.debug.print("\nslow (@exp2): {d:.3} ns/code\n", .{timeE2m1(io, e2m1ToF32SimdSlow, blocks, out)});
    std.debug.print("lut: {d:.3} ns/code\n", .{timeE2m1(io, e2m1ToF32SimdLut, blocks, out)});
    std.debug.print("bits: {d:.3} ns/code\n", .{timeE2m1(io, e2m1ToF32Simd, blocks, out)});
}

// Start bit for E4M3 is 7
// 3 for exponent and 0 for mantissa

// s·2^7 + e·2^3 + m·2^0 for seeeemmm
// s′·2^31 + E′·2^23 + M′ for f32
// Normal (e != 0): (−1)^s · 2^(e−7) · (1 + m/8)
// (−1)^s′ = (−1)^s gives s′ = s, the sign moves from bit 7 to bit 31, so shift by 24
// 2^(E′−127) = 2^(e−7) gives E′ = e + 120
// M′/2^23 = m/8 gives M′ = m·2^20

// We could do the lut again as in
// pierrelgol/nvfp4-dequantizer
// but i really want to find out the bit shifting and bit masking way

// Paper from vLLM: "Who Says Elephants Can't Run" (Kim et al., 2022, arXiv:2211.10017).
// They entirely discard the nans to make the code run faster

// https://github.com/Maratyszcza/FP16/blob/master/include/fp16/fp16.h
// I'm going to take this implementation as reference

// https://www.rapidtables.com/convert/number/hex-to-binary.html
// My beloved hex to bin converter

pub fn e4m3ToF32bits(bits: @Vector(16, u8)) @Vector(16, f32) {
    const V = @Vector(16, u32);
    const F = @Vector(16, f32);
    // Widen to u32 lanes, so every field can move to its place in the f32
    const wide: V = bits;
    // Only keep the first sign bit, then shift it to be at its final position in the f32
    // signs are equal
    const sign = (wide & @as(V, @splat(0x80))) << @splat(24);
    // Now, we recover the exponent and mantissa, we drop the sign and
    // we perform (e<<23)|(m<<20) at the same time, so we can add the bias of 120 to the exponent
    const expMant = (wide & @as(V, @splat(0x7F))) << @splat(20);

    // We then fix the bias by adding 120 so that
    // E' = e + 120 (cf the top computations)
    const normal = expMant + @as(V, @splat(120 << 23));
    // We substract the smallest normal to get the subnormal
    const smallestNormal: F = @splat(0x1p-6);
    // This is where i'm starting to get confused by Maratyszcza's code,
    // We technically build a fake normal number, we place the mantissa
    // as a normal number
    // it's their "magic_mask magic_bias"
    // adding 121 << 23 gives 2^-6 · (1 + m/8) which is exactly what we need for subnormal
    const built: F = @bitCast(expMant + @as(V, @splat(121 << 23)));
    // Substracting this "fake number" gives the right result (sterbenz lemma)
    // https://en.wikipedia.org/wiki/Sterbenz_lemma
    // tldr x - y is only equal if y/2 <= x <= 2y, which is the case here,
    // as we are in the subnormal range
    // That's a neat trick
    const subnormal: V = @bitCast(built - smallestNormal);

    // Check if we're normal or subnormal
    // 0x78 = 0 1111 000 hence only the exponent bits are checked, if they are all 0, then it's subnormal
    const isSubnormal = (wide & @as(V, @splat(0x78))) == @as(V, @splat(0));
    // Nan is only appearing when S 1111 111, hence only in the case we are normal
    const finite = @select(u32, isSubnormal, subnormal, normal);

    // if ((bits & 0x7F) == 0x7F) return math.nan(f32);
    const isNan = (wide & @as(V, @splat(0x7F))) == @as(V, @splat(0x7F));
    const magnitude = @select(u32, isNan, @as(V, @splat(0x7FC00000)), finite);

    // The sign is the top bit of the float: 0 with the sign set is -0.0
    return @bitCast(magnitude | sign);
}

test "E4M3 decodes the scalar cases as one block" {
    // The scalar cases are exactly 16 codes, so they fill one block of 16 lanes
    comptime std.debug.assert(e4m3Cases.len == 16);
    var bits: [16]u8 = undefined;
    var want: [16]f32 = undefined;

    // reuse the scalar cases to fill a block of 16 lanes, so we can test the SIMD implementation
    for (e4m3Cases, 0..) |c, i| {
        bits[i] = c.bits;
        want[i] = c.want;
    }
    const got: [16]f32 = e4m3ToF32bits(bits);
    for (bits, want, got) |b, w, g| {
        if (math.isNan(w)) {
            std.testing.expect(math.isNan(g)) catch |err| {
                std.debug.print("failed on 0x{X}\n", .{b});
                return err;
            };
            continue;
        }
        // Compare bits so -0.0 and 0.0 stay different
        std.testing.expectEqual(@as(u32, @bitCast(w)), @as(u32, @bitCast(g))) catch |err| {
            std.debug.print("failed on 0x{X}: got {d}\n", .{ b, g });
            return err;
        };
    }
}

test "E4M3 NaN treatment on simd" {
    const V = @Vector(16, u32);

    // The two NaN codes, mixed with near misses: all exponent bits, or all mantissa bits, but not both
    const bits: @Vector(16, u8) = .{ 0x7F, 0xFF, 0x7E, 0xFE, 0x78, 0xF8, 0x77, 0xF7, 0x3F, 0xBF, 0x00, 0x80, 0x7F, 0xFF, 0x07, 0x87 };
    const wantNan = [16]bool{ true, true, false, false, false, false, false, false, false, false, false, false, true, true, false, false };

    const wide: V = bits;
    // Mask the sign off so both 0x7F and 0xFF count, every lane gets its own true or false
    const isNan = (wide & @as(V, @splat(0x7F))) == @as(V, @splat(0x7F));
    // Stand-in for the decoded values, the NaN step does not care what is in there, it will be masked out anyway, so we can just use 0.0 for the finite lanes
    const finite: V = @splat(0);
    // No early return in SIMD: each lane picks the f32 NaN pattern or keeps its value
    // just mask the sign to either nan or finite
    const outV: @Vector(16, f32) = @bitCast(@select(u32, isNan, @as(V, @splat(0x7FC00000)), finite));

    const codes: [16]u8 = bits;
    const out: [16]f32 = outV;
    for (codes, wantNan, out) |b, w, g| {
        std.testing.expectEqual(w, math.isNan(g)) catch |err| {
            std.debug.print("failed on 0x{X}: got {d}\n", .{ b, g });
            return err;
        };
    }
}
