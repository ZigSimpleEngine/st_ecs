//! P9: ctz-пилинг только нужной стороны (побайтово, как p8).
//! Активные: пока есть единицы — ctz, вызов, сброс младшей единицы.
//! Неактивные зеркально через ~byte. Противоположные биты не посещаются
//! вообще: ни веток на них, ни вызовов.
//! Зависимостей кроме std нет:
//!   zig test src/iteration_tests/p9.zig
const std = @import("std");

pub fn iterateActiveByte(byte: u8, comptime on_active: fn (u3) void) void {
    var w = byte;
    while (w != 0) {
        const i: u32 = @ctz(w);
        on_active(@truncate(i));
        w &= w - 1;
    }
}

pub fn iterateInactiveByte(byte: u8, comptime on_inactive: fn (u3) void) void {
    var w: u8 = ~byte;
    while (w != 0) {
        const i: u32 = @ctz(w);
        on_inactive(@truncate(i));
        w &= w - 1;
    }
}

/// Тот же пилинг целым словом u64: один цикл на 64 бита вместо 8 побайтовых.
/// Индекс бита 0..63.
pub fn iterateActiveWord(word: u64, comptime on_active: fn (u6) void) void {
    var w = word;
    while (w != 0) {
        const i: u32 = @ctz(w);
        on_active(@truncate(i));
        w &= w - 1;
    }
}

pub fn iterateInactiveWord(word: u64, comptime on_inactive: fn (u6) void) void {
    var w: u64 = ~word;
    while (w != 0) {
        const i: u32 = @ctz(w);
        on_inactive(@truncate(i));
        w &= w - 1;
    }
}

// ================= Тесты: только p9 =================

var g_seen: [8]bool = undefined;
var g_count: u32 = 0;

fn tA(bit: u3) void {
    g_seen[bit] = true;
    g_count += 1;
}

fn tI(bit: u3) void {
    g_seen[bit] = true;
    g_count += 1;
}

test "p9 active: all 256 bytes hit exactly the set bits" {
    const t = std.testing;
    var v: u32 = 0;
    while (v < 256) : (v += 1) {
        const b: u8 = @truncate(v);
        g_seen = [_]bool{false} ** 8;
        iterateActiveByte(b, tA);
        var i: u32 = 0;
        while (i < 8) : (i += 1) {
            try t.expectEqual(((v >> @truncate(i)) & 1) == 1, g_seen[i]);
        }
    }
}

test "p9 inactive: all 256 bytes hit exactly the clear bits" {
    const t = std.testing;
    var v: u32 = 0;
    while (v < 256) : (v += 1) {
        const b: u8 = @truncate(v);
        g_seen = [_]bool{false} ** 8;
        iterateInactiveByte(b, tI);
        var i: u32 = 0;
        while (i < 8) : (i += 1) {
            try t.expectEqual(((v >> @truncate(i)) & 1) == 0, g_seen[i]);
        }
    }
}

test "p9 sums over full u16 range (+=1 per call)" {
    const t = std.testing;
    g_count = 0;
    var w: u32 = 0;
    while (w < 65536) : (w += 1) {
        iterateActiveByte(@truncate(w), tA);
        iterateActiveByte(@truncate(w >> 8), tA);
    }
    try t.expectEqual(@as(u32, 524288), g_count);
    g_count = 0;
    w = 0;
    while (w < 65536) : (w += 1) {
        iterateInactiveByte(@truncate(w), tI);
        iterateInactiveByte(@truncate(w >> 8), tI);
    }
    try t.expectEqual(@as(u32, 524288), g_count);
}

// ABI-стабильная точка для asm-дампов (шаг asm-iter) и внешних замеров.
var g_bench: u64 = 0;

fn benchA(bit: u3) void {
    _ = bit;
    g_bench += 1;
}

export fn p9_countU16(w: u16) u64 {
    g_bench = 0;
    iterateActiveByte(@truncate(w), benchA);
    iterateActiveByte(@truncate(w >> 8), benchA);
    return g_bench;
}

// ================= Тесты словых версий (u64) =================

var g_seen64: [64]bool = undefined;

fn tA64(bit: u6) void {
    g_seen64[bit] = true;
}

fn tI64(bit: u6) void {
    g_seen64[bit] = true;
}

fn checkWord64(w: u64, active: bool) !void {
    const t = std.testing;
    g_seen64 = [_]bool{false} ** 64;
    if (active) {
        iterateActiveWord(w, tA64);
    } else {
        iterateInactiveWord(w, tI64);
    }
    var i: u32 = 0;
    while (i < 64) : (i += 1) {
        const bit_set = ((w >> @truncate(i)) & 1) == 1;
        if (active) {
            try t.expectEqual(bit_set, g_seen64[i]);
        } else {
            try t.expectEqual(!bit_set, g_seen64[i]);
        }
    }
}

test "p9 word: edges u64, each bit exactly once" {
    const edges = [_]u64{
        0, std.math.maxInt(u64), 1, 1 << 63,
        0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555,
        0xFF00_FF00_FF00_FF00, 0x00FF_00FF_00FF_00FF,
        0x0000_0000_0000_00FF, 0xFF00_0000_0000_0000,
        0x0000_FFFF_0000_FFFF, 0xFFFF_0000_FFFF_0000,
    };
    for (edges) |w| {
        try checkWord64(w, true);
        try checkWord64(w, false);
    }
    // Псевдослучайная выборка тем же xorshift, что в common (сид тот же).
    var s: u64 = 0x9E3779B97F4A7C15;
    var k: usize = 0;
    while (k < 5000) : (k += 1) {
        var x = s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        s = x;
        const w = x *% 0x2545F4914F6CDD1D;
        try checkWord64(w, true);
        try checkWord64(w, false);
    }
}
