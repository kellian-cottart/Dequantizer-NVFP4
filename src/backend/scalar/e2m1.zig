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
