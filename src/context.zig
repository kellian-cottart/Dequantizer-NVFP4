// const std = @import("std");
// const fc = @import("formatConversion.zig");

// const Backend = enum {
//     scalar,
//     simd,
// };

// // Struct to hold the context, such that we can use a decode fct
// // that selects between scalar and simd implementations?
// pub const Nvfp4Context = struct {
//     scales: []const u8,
//     scale_2: f32,
//     rows: usize,
//     cols: usize,

//     pub const block_bytes = 8;

//     pub fn init(scales: []const u8, scale_2: f32, rows: usize, cols: usize) error{InvalidShape}!Nvfp4Context {
//         if (cols % 16 != 0) {
//             return error.InvalidShape;
//         }
//         if (rows * cols / 2 != scales.len * block_bytes) {
//             return error.InvalidShape;
//         }
//         return Nvfp4Context{
//             .scales = scales,
//             .scale_2 = scale_2,
//             .rows = rows,
//             .cols = cols,
//         };
//     }

//     pub fn decode(ctx: Nvfp4Context, backend: Backend, first_byte: usize, chunk: []const u8, out: []f32) void {
//         std.debug.assert(first_byte % block_bytes == 0);
//         std.debug.assert(chunk.len % block_bytes == 0);
//         std.debug.assert(out.len == 2 * chunk.len);
//         const scales = ctx.scales[first_byte / block_bytes ..][0 .. chunk.len / block_bytes];
//         switch (backend) {
//             .scalar => decodeScalar(chunk, scales, ctx.scale_2, out),
//             .simd => decodeSimd(chunk, scales, ctx.scale_2, out),
//         }
//     }
// };

// pub fn decodeScalar()
