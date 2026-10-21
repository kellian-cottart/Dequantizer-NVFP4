const std = @import("std");
const format = @import("Dequantizer").formatConversion;

const Backend = enum {
    scalar,
    simd,
};
