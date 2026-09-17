//! P8: прямой вызов на бит, без длин и склейки вообще.
//! Большой switch через `inline else`: значение байта comptime-известно,
//! каждая ветка — 8 последовательных вызовов, выбор колбека — comptime.
//! inactive разделяются от active самим фактом вызова:
//!   on_inactive(byte_id, bit) / on_active(bit).
//! Зависимостей кроме std нет — файл тестируется напрямую:
//!   zig test src/iteration_tests/p8.zig
const std = @import("std");

pub fn iterateByte(
    byte_id: u32,
    byte: u8,
    comptime on_inactive: ?fn (u32, u3) void,
    comptime on_active: ?fn (u3) void,
) void {
    @setEvalBranchQuota(100_000);
    switch (byte) {
        inline else => |b| {
            inline for (0..8) |i| {
                if (((b >> i) & 1) == 1) {
                    if (on_active) |f| f(@truncate(i));
                } else {
                    if (on_inactive) |f| f(byte_id, @truncate(i));
                }
            }
        },
    }
}

// ================= Тест: только p8, без общих зависимостей =================

var g_seen_a: [8]bool = undefined;
var g_seen_i: [8]bool = undefined;
var g_count_a: u32 = 0;
var g_count_i: u32 = 0;
var g_exp_byte: u32 = 0;

fn tActive(bit: u3) void {
    g_seen_a[bit] = true;
    g_count_a += 1;
}

fn tInactive(byte_id: u32, bit: u3) void {
    std.debug.assert(byte_id == g_exp_byte);
    g_seen_i[bit] = true;
    g_count_i += 1;
}

test "p8: all 256 bytes, every bit called exactly once on the right side" {
    const t = std.testing;
    const byte_ids = [_]u32{ 0, 1, 7, 12345, 0xFFFF_FFFF };
    for (byte_ids) |bid| {
        var v: u32 = 0;
        while (v < 256) : (v += 1) {
            const b: u8 = @truncate(v);
            g_seen_a = [_]bool{false} ** 8;
            g_seen_i = [_]bool{false} ** 8;
            g_exp_byte = bid;
            iterateByte(bid, b, tInactive, tActive);
            var i: u32 = 0;
            while (i < 8) : (i += 1) {
                const want_active = ((v >> @truncate(i)) & 1) == 1;
                if (want_active) {
                    try t.expect(g_seen_a[i] and !g_seen_i[i]);
                } else {
                    try t.expect(g_seen_i[i] and !g_seen_a[i]);
                }
            }
        }
    }
}

test "p8: call sums over full u16 range (+=1 per call)" {
    const t = std.testing;
    g_count_a = 0;
    g_count_i = 0;
    var w: u32 = 0;
    while (w < 65536) : (w += 1) {
        const lo: u8 = @truncate(w);
        const hi: u8 = @truncate(w >> 8);
        g_exp_byte = w * 2;
        iterateByte(w * 2, lo, tInactive, tActive);
        g_exp_byte = w * 2 + 1;
        iterateByte(w * 2 + 1, hi, tInactive, tActive);
    }
    // Симметрия полного перебора: половина из 1048576 вызовов — active.
    try t.expectEqual(@as(u32, 524288), g_count_a);
    try t.expectEqual(@as(u32, 524288), g_count_i);
}

// ABI-стабильная точка для asm-дампов (шаг asm-iter) и внешних замеров.
var g_bench_a: u64 = 0;
var g_bench_i: u64 = 0;

fn benchA(bit: u3) void {
    _ = bit;
    g_bench_a += 1;
}

fn benchI(byte_id: u32, bit: u3) void {
    _ = byte_id;
    _ = bit;
    g_bench_i += 1;
}

export fn p8_countU16(w: u16) u64 {
    const lo: u8 = @truncate(w);
    const hi: u8 = @truncate(w >> 8);
    iterateByte(0, lo, benchI, benchA);
    iterateByte(1, hi, benchI, benchA);
    return (g_bench_a << 32) | g_bench_i;
}
