const std = @import("std");

pub const MaxInt64: u64 = std.math.maxInt(u64);
pub const List = std.ArrayList;

/// List with alignment to 64 bytes
pub fn ListA64(comptime T: type) type {
    return std.array_list.Aligned(T, std.mem.Alignment.@"64");
}
