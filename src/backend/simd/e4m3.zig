const std = @import("std");
const math = std.math;

const e4m3Cases = @import("../scalar/e4m3.zig").e4m3Cases;

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

// The scales are independent, so the lane count n is free: comptime builds one version per width used
pub fn e4m3ToF32bits(comptime n: usize, bits: @Vector(n, u8)) @Vector(n, f32) {
    const V = @Vector(n, u32);
    const F = @Vector(n, f32);
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
    const got: [16]f32 = e4m3ToF32bits(16, bits);
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
