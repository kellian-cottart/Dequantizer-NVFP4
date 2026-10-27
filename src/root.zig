pub const backend = struct {
    pub const scalar = struct {
        pub const e2m1 = @import("backend/scalar/e2m1.zig");
        pub const e4m3 = @import("backend/scalar/e4m3.zig");
        pub const multiply = @import("backend/scalar/multiply.zig");
    };
    pub const simd = struct {
        pub const e2m1 = @import("backend/simd/e2m1.zig");
        pub const e4m3 = @import("backend/simd/e4m3.zig");
        pub const multiply = @import("backend/simd/multiply.zig");
    };
};
pub const read = @import("read.zig");
pub const dequantizer = @import("dequantizer.zig");

test {
    _ = backend.scalar.e2m1;
    _ = backend.scalar.e4m3;
    _ = backend.scalar.multiply;
    _ = backend.simd.e2m1;
    _ = backend.simd.e4m3;
    _ = backend.simd.multiply;
    _ = read;
    _ = dequantizer;
}
