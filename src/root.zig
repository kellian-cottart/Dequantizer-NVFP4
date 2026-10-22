pub const backend = struct {
    pub const scalar = @import("backend/scalar.zig");
    pub const simd = @import("backend/simd.zig");
};
pub const read = @import("read.zig");
pub const dequantizer = @import("dequantizer.zig");

test {
    _ = backend.scalar;
    _ = backend.simd;
    _ = read;
    _ = dequantizer;
}
