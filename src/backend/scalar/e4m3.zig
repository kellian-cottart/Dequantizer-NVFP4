const std = @import("std");
const math = std.math;

// Test tables shared by the scalar and simd backends

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

/// Function E4M3
/// s eeee mmm
// Normal (`e != 0`): `(-1)^s * 2^(e - 7) * (1 + m/8)`
// Subnormal (`e == 0`): `(-1)^s * 2^(-6) * (m/8)`
pub fn e4m3ToF32(bits: u8) !f32 {
    // ​S.1111.111 => NaN
    // Mask the sign off so both 0x7F and 0xFF count
    if ((bits & 0x7F) == 0x7F) return math.nan(f32);
    const sign = (bits >> 7) & 1;
    const exponent = (bits >> 3) & 0b1111;
    const mantissa = bits & 0b111;
    const floatMantissa: f32 = @floatFromInt(mantissa);
    const floatMagnitude: f32 = if (exponent != 0) (1.0 + floatMantissa / 8.0) * math.pow(f32, 2.0, @as(f32, @floatFromInt(exponent)) - 7.0) else (1.0 * floatMantissa / 8.0) * math.pow(f32, 2.0, -6.0);
    return if (sign == 1) -floatMagnitude else floatMagnitude;
}

test "E4M3 decodes cases" {
    for (e4m3Cases) |c| {
        const got = try e4m3ToF32(c.bits);
        // NaN != NaN, so expectEqual can never pass on it
        if (math.isNan(c.want)) {
            std.testing.expect(math.isNan(got)) catch |err| {
                std.debug.print("failed on 0x{X}\n", .{c.bits});
                return err;
            };
            continue;
        }
        std.testing.expectEqual(c.want, got) catch |err| {
            std.debug.print("failed on 0x{X}\n", .{c.bits});
            return err;
        };
    }
}
