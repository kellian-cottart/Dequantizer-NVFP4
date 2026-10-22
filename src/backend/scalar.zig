const std = @import("std");
const Io = std.Io;
const math = std.math;

const e2m1Cases = @import("cases.zig").e2m1Cases;
const e4m3Cases = @import("cases.zig").e4m3Cases;
const nvfp4Cases = @import("cases.zig").nvfp4Cases;

/// Function E2M1
/// the input is format SEEM (Sign, Exponent, Exponent, Mantissa)
/// the output is a float32
/// Float Value = E2M1_value × FP8_block_scale × FP32_global_scale
pub fn e2m1ToF32(bits: u4) !f32 {
    // Read only the first bit
    const sign = (bits >> 3) & 1;
    // Shift to the right, and with 0b11 (0011) (remove the sign bit to only get the 2 middle bits)
    const exponent = (bits >> 1) & 0b11;
    // Mantissa is just the last bit
    const mantissa = bits & 1;
    // Convert the mantissa to its floating representation
    const floatMantissa: f32 = @floatFromInt(mantissa);
    // Magnitude is the bits to the power
    const floatMagnitude: f32 = if (exponent == 0) 0.5 * floatMantissa else (1 + floatMantissa / 2) * math.pow(f32, 2.0, @as(f32, @floatFromInt(exponent)) - 1.0);
    return if (sign == 1) -floatMagnitude else floatMagnitude;
}

test "E2M1 decodes all 16 codes" {
    for (e2m1Cases) |c| {
        std.testing.expectEqual(c.want, try e2m1ToF32(c.bits)) catch |err| {
            std.debug.print("failed on 0x{X}\n", .{c.bits});
            return err;
        };
    }
}

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
