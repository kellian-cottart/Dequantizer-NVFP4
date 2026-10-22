const std = @import("std");
const math = std.math;

// Test tables shared by the scalar and simd backends

// E2M1 actually has 16 codes, a full LUT is available
// https://binarycon.com/tools/nvfp4-converter-analyzer/
pub const e2m1Cases = [_]struct { bits: u4, want: f32 }{
    .{ .bits = 0x0, .want = 0.0 }, // 0000
    .{ .bits = 0x1, .want = 0.5 }, // 0001
    .{ .bits = 0x2, .want = 1.0 }, // 0010
    .{ .bits = 0x3, .want = 1.5 }, // 0011
    .{ .bits = 0x4, .want = 2.0 }, // 0100
    .{ .bits = 0x5, .want = 3.0 }, // 0101
    .{ .bits = 0x6, .want = 4.0 }, // 0110
    .{ .bits = 0x7, .want = 6.0 }, // 0111
    .{ .bits = 0x8, .want = -0.0 }, // 1000
    .{ .bits = 0x9, .want = -0.5 }, // 1001
    .{ .bits = 0xA, .want = -1.0 }, // 1010
    .{ .bits = 0xB, .want = -1.5 }, // 1011
    .{ .bits = 0xC, .want = -2.0 }, // 1100
    .{ .bits = 0xD, .want = -3.0 }, // 1101
    .{ .bits = 0xE, .want = -4.0 }, // 1110
    .{ .bits = 0xF, .want = -6.0 }, // 1111
};

// E4M3 has 256 codes, hence i took a few representative ones, including the special case of NaN
// Values cross-checked against ml_dtypes.float8_e4m3fn
// Asked Claude Code to generate it
pub const e4m3Cases = [_]struct { bits: u8, want: f32 }{
    .{ .bits = 0x00, .want = 0.0 },
    .{ .bits = 0x01, .want = 0.001953125 }, // smallest subnormal, 2^-9
    .{ .bits = 0x07, .want = 0.013671875 }, // largest subnormal, 7/8 * 2^-6
    .{ .bits = 0x08, .want = 0.015625 }, // smallest normal, 2^-6
    .{ .bits = 0x09, .want = 0.017578125 }, // smallest normal + 1 mantissa step
    .{ .bits = 0x38, .want = 1.0 },
    .{ .bits = 0x3F, .want = 1.875 }, // all mantissa bits set
    .{ .bits = 0x78, .want = 256.0 }, // exponent 15 is still a normal, not inf
    .{ .bits = 0x7D, .want = 416.0 }, // exponent 15 with mantissa, below the NaN code
    .{ .bits = 0x7E, .want = 448.0 }, // max
    .{ .bits = 0x7F, .want = math.nan(f32) }, // special case: NaN
    .{ .bits = 0x80, .want = -0.0 },
    .{ .bits = 0x81, .want = -0.001953125 }, // negative subnormal
    .{ .bits = 0xB8, .want = -1.0 },
    .{ .bits = 0xFE, .want = -448.0 }, // min
    .{ .bits = 0xFF, .want = math.nan(f32) }, // negative NaN
};

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
