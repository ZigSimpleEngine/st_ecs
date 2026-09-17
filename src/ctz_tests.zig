const std = @import("std");
const print = std.debug.print;
const t = std.testing;
const BitWord = @import("bit_word.zig").BitWord;

test "ctz" {
    const it = Iterator(u8, printActiveLen, printInactiveLen);
    const in = [_]u8{ 0b0010_0010, 0b1000_0001, 0b1111_1100, 0b1111_0000, 0b0000_0001, 0b1111_1111 };

    for (in) |value| {
        print("\nWord = {}\n", .{value});
        it.iterate_word(value);
    }
}

fn printActiveLen(len: u32) void {
    print("active len = {}\n", .{len});
}

fn printInactiveLen(len: u32) void {
    print("inactive len = {}\n", .{len});
}

fn Iterator(comptime Word: type, comptime on_active: ?fn (len: u32) void, comptime on_inactive: ?fn (len: u32) void) type {
    const bw = BitWord(Word);
    return struct {
        fn iterate_word(word: Word) void {
            const start_bit_value: u1 = @truncate(word & 1);
            var word_mut = if (start_bit_value == 0) word else ~word;
            var old_bit_counter: u32 = 0;
            var bit_counter: u32 = 0;
            var ctz: u32 = 0;
            var active_flag = start_bit_value;

            while (true) {
                ctz = @ctz(word_mut);
                old_bit_counter = bit_counter;
                bit_counter += ctz;
                // print("word_mut = {b:08}\n", .{word_mut});

                if (bit_counter >= bw.word_type_bits) {
                    if (active_flag == 0) {
                        if (on_inactive) |f| f(bw.word_type_bits - old_bit_counter);
                    } else {
                        if (on_active) |f| f(bw.word_type_bits - old_bit_counter);
                    }
                    break;
                } else {
                    if (active_flag == 0) {
                        if (on_inactive) |f| f(ctz);
                    } else {
                        if (on_active) |f| f(ctz);
                    }
                }

                active_flag +%= 1;
                word_mut = ~word_mut;
                word_mut >>= @truncate(ctz);
                word_mut &= bw.maskEndInverted(@truncate(bit_counter));
            }
        }
    };
}

// --- Oracle autotest: сверяет РЕАЛЬНЫЙ iterate_word с побитовым эталоном ---
// Колбэки Iterator не принимают контекст, поэтому коллектор — через globals.
var g_active: [64]u32 = undefined;
var g_active_len: usize = 0;
var g_inactive: [64]u32 = undefined;
var g_inactive_len: usize = 0;

fn collectActive(len: u32) void {
    g_active[g_active_len] = len;
    g_active_len += 1;
}

fn collectInactive(len: u32) void {
    g_inactive[g_inactive_len] = len;
    g_inactive_len += 1;
}

fn oracleRuns(comptime Word: type, word: Word, exp_a: *[64]u32, exp_i: *[64]u32) struct { a_len: usize, i_len: usize } {
    const bw = BitWord(Word);
    var cur: u1 = @truncate(word & 1);
    var len: u32 = 0;
    var a_len: usize = 0;
    var i_len: usize = 0;
    var b: u32 = 0;
    while (b < bw.word_type_bits) : (b += 1) {
        const bit: u1 = @truncate((word >> @truncate(b)) & 1);
        if (bit == cur) {
            len += 1;
        } else {
            if (cur == 1) {
                exp_a[a_len] = len;
                a_len += 1;
            } else {
                exp_i[i_len] = len;
                i_len += 1;
            }
            cur = bit;
            len = 1;
        }
    }
    if (cur == 1) {
        exp_a[a_len] = len;
        a_len += 1;
    } else {
        exp_i[i_len] = len;
        i_len += 1;
    }
    return .{ .a_len = a_len, .i_len = i_len };
}

fn checkWord(comptime Word: type, word: Word) !void {
    const bw = BitWord(Word);
    const It = Iterator(Word, collectActive, collectInactive);
    var exp_a: [64]u32 = undefined;
    var exp_i: [64]u32 = undefined;

    g_active_len = 0;
    g_inactive_len = 0;
    It.iterate_word(word);
    const o = oracleRuns(Word, word, &exp_a, &exp_i);

    try t.expectEqualSlices(u32, exp_a[0..o.a_len], g_active[0..g_active_len]);
    try t.expectEqualSlices(u32, exp_i[0..o.i_len], g_inactive[0..g_inactive_len]);

    var total: u32 = 0;
    for (g_active[0..g_active_len]) |x| total += x;
    for (g_inactive[0..g_inactive_len]) |x| total += x;
    try t.expectEqual(bw.word_type_bits, total);
}

test "ctz iterate_word vs oracle: exhaustive u8" {
    var v: u32 = 0;
    while (v < 256) : (v += 1) {
        try checkWord(u8, @truncate(v));
    }
}

test "ctz iterate_word vs oracle: edges u16/u32/u64" {
    const cases16 = [_]u16{ 0, 0xFFFF, 1, 0x8000, 0xAAAA, 0x5555, 0x00FF, 0xFF00, 0b1010 };
    for (cases16) |w| {
        try checkWord(u16, w);
    }
    const cases32 = [_]u32{ 0, 0xFFFF_FFFF, 1, 0x8000_0000, 0xAAAA_AAAA, 0x5555_5555, 0x00FF_FF00, 0xFFFF_0000 };
    for (cases32) |w| {
        try checkWord(u32, w);
    }
    const cases64 = [_]u64{ 0, std.math.maxInt(u64), 1, 1 << 63, 0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555, 0xFFFF_FFFF_0000_0000, 0x0000_0000_FFFF_FFFF, 0b1010, 0x8000_0000_0000_0001 };
    for (cases64) |w| {
        try checkWord(u64, w);
    }
}
