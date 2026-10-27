const std = @import("std");
const math = std.math;

const e2m1ToF32 = @import("e2m1.zig").e2m1ToF32;
const e4m3ToF32 = @import("e4m3.zig").e4m3ToF32;

// Test tables shared by the scalar and simd backends

// NVFP4 is a combination of E2M1 and E4M3, with an additional F32 scale factor. The final value is computed as:
// value = e2m1 * e4m3_block_scale * f32_tensor_scale
// There are many cases, but we will test a few representative ones, including the special case of NaN
// Asked claude code to generate it
pub const nvfp4Cases = [_]struct { weights: u8, scale: u8, scale2: f32, want: f32 }{
    .{ .weights = 0x02, .scale = 0x38, .scale2 = 1.0, .want = 1.0 }, // identity
    .{ .weights = 0x0F, .scale = 0x38, .scale2 = 1.0, .want = -6.0 }, // e2m1 sign comes through
    .{ .weights = 0x03, .scale = 0x3F, .scale2 = 0.25, .want = 0.703125 }, // mantissa bits in all three
    .{ .weights = 0x07, .scale = 0x7E, .scale2 = 1.0, .want = 2688.0 }, // largest block magnitude
    .{ .weights = 0x01, .scale = 0x01, .scale2 = 1.0, .want = 0.0009765625 }, // smallest nonzero, subnormal scale
    .{ .weights = 0x02, .scale = 0x38, .scale2 = 0.1, .want = 0.1 }, // non-dyadic tensor scale passes through untouched
    .{ .weights = 0x0B, .scale = 0xB8, .scale2 = 1.0, .want = 1.5 }, //  a negative block scale flips the sign
    .{ .weights = 0x08, .scale = 0x38, .scale2 = 1.0, .want = -0.0 }, // -0 stays -0
    .{ .weights = 0x72, .scale = 0x38, .scale2 = 1.0, .want = 1.0 }, // discarding the upper nibble of weights, only the lower nibble is used
    .{ .weights = 0x00, .scale = 0x7F, .scale2 = 1.0, .want = math.nan(f32) }, // 0 * NaN scale is NaN, not 0
};

/// In a ModelOpt-style checkpoint:
/// - `X.weight`: `U8`, shape `[rows, cols/2]` (packed nibbles)
/// - `X.weight_scale`: `F8_E4M3`, shape `[rows, cols/16]`
/// - `X.weight_scale_2`: `F32`, scalar
/// Blocks of **16** E2M1 values sharing one **E4M3** scale, plus one **F32** scale per tensor
/// `value = e2m1 * e4m3_block_scale * f32_tensor_scale`
pub fn oneNvfp4ToF32(weights: u8, weightScale: u8, weightScale2: f32) !f32 {
    const e2m1Value = try e2m1ToF32(@as(u4, @truncate(weights)));
    const e4m3Value = try e4m3ToF32(weightScale);
    return e2m1Value * e4m3Value * weightScale2;
}

test "NVFP4 decodes cases" {
    for (nvfp4Cases) |c| {
        const got = try oneNvfp4ToF32(c.weights, c.scale, c.scale2);
        if (math.isNan(c.want)) {
            std.testing.expect(math.isNan(got)) catch |err| {
                std.debug.print("failed on weights=0x{X} scale=0x{X}\n", .{ c.weights, c.scale });
                return err;
            };
            continue;
        }
        // I'm casting to u32 allows to compare -0.0 and 0.0 because they have different bit patterns, but are equal in f32
        // -0.0 is important to know: 1/0.0 is +inf, while 1/-0.0 is -inf
        std.testing.expectEqual(@as(u32, @bitCast(c.want)), @as(u32, @bitCast(got))) catch |err| {
            std.debug.print("failed on weights=0x{X} scale=0x{X}: got {d}\n", .{ c.weights, c.scale, got });
            return err;
        };
    }
}
