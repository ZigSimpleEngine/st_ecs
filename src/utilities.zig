const std = @import("std");

pub const MaxInt64: u64 = std.math.maxInt(u64);
pub const List = std.ArrayList;

/// List with alignment to 64 bytes
pub fn ListA64(comptime T: type) type {
    return std.array_list.Aligned(T, std.mem.Alignment.@"64");
}

pub fn IteratorCallback(Context: type) type {
    return ?fn (context: Context, bit_id: u32) bool;
}

pub fn InlineIteratorCallback(Context: type) type {
    return ?fn (context: Context, bit_id: u32) callconv(.@"inline") bool;
}

pub fn assertUnsignedPowerOfTwoInt(comptime T: type) void {
    const info = @typeInfo(T);

    switch (info) {
        .int => |int_info| {
            if (int_info.signedness != .unsigned) {
                @compileError("expected an unsigned integer type");
            }

            const bits = int_info.bits;

            if (bits < 2 or (bits & (bits - 1)) != 0) {
                @compileError("unsigned integer bit size must be a power of two >= 2");
            }
        },
        else => @compileError("expected an unsigned integer type"),
    }
}

pub inline fn iterateActiveBitsInWord(
    comptime Word: type,
    comptime Context: type,
    comptime callback: fn (context: Context, bit_id: u32) bool,
    context: Context,
    start_bit_id: u32,
    word: Word,
) bool {
    comptime {
        assertUnsignedPowerOfTwoInt(Word);
    }

    var w = word;
    while (w != 0) {
        const i: u32 = @ctz(w);
        if (!callback(context, start_bit_id + i)) return false;
        w &= w - 1;
    }
    return true;
}

pub inline fn iterateActiveBitsInWordInline(
    comptime Word: type,
    comptime Context: type,
    comptime callback: fn (context: Context, bit_id: u32) callconv(.@"inline") bool,
    context: Context,
    start_bit_id: u32,
    word: Word,
) bool {
    comptime {
        assertUnsignedPowerOfTwoInt(Word);
    }

    var w = word;
    while (w != 0) {
        const i: u32 = @ctz(w);
        if (!callback(context, start_bit_id + i)) return false;
        w &= w - 1;
    }
    return true;
}

pub const BitRange = struct { start: u32, len: u32, layer: u32 };

pub const BitState = enum(u1) {
    const Self = @This();

    inactive = 0,
    active = 1,

    pub inline fn toWordState(self: Self, comptime Word: type) Word {
        comptime {
            assertUnsignedPowerOfTwoInt(Word);
        }
        return 0 -% @as(Word, @intFromEnum(self));
    }

    pub inline fn asBool(self: Self) bool {
        return self == .active;
    }
};
