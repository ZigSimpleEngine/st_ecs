//! P3: comptime LUT всех 65536 слов u16.
//! Запись: длины ранов + старт + готовые суммы. Запрос — lookup + 2 сложения,
//! ноль ctz и ноль веток на ран. Разделение типов важно — оно закодировано
//! в start (раны строго чередуются), суммы предподсчитаны.
const std = @import("std");
const common = @import("common.zig");
const W = common.W;

pub const Entry = struct {
    lens: [16]u8 = undefined,
    n: u8 = 0,
    start: u1 = 0,
    sum_a: u16 = 0,
    sum_i: u16 = 0,
};

const table: [common.N]Entry = blk: {
    @setEvalBranchQuota(10_000_000);
    var t: [common.N]Entry = undefined;
    var v: u32 = 0;
    while (v < common.N) : (v += 1) {
        const w: W = @truncate(v);
        var e = Entry{};
        var cur: u1 = @truncate(w & 1);
        e.start = cur;
        var len: u32 = 0;
        var b: u32 = 0;
        while (b < common.BITS) : (b += 1) {
            const bit: u1 = @truncate((w >> @truncate(b)) & 1);
            if (bit == cur) {
                len += 1;
            } else {
                e.lens[e.n] = @truncate(len);
                e.n += 1;
                if (cur == 1) {
                    e.sum_a += @truncate(len);
                } else {
                    e.sum_i += @truncate(len);
                }
                cur = bit;
                len = 1;
            }
        }
        e.lens[e.n] = @truncate(len);
        e.n += 1;
        if (cur == 1) {
            e.sum_a += @truncate(len);
        } else {
            e.sum_i += @truncate(len);
        }
        t[v] = e;
    }
    break :blk t;
};

pub fn count(word: W, sums: *common.Sums) void {
    const e = table[word];
    sums.active += e.sum_a;
    sums.inactive += e.sum_i;
}

/// Полные длины с разделением типов (для потребителей и тестов).
pub fn runs(word: W, out: *common.Runs) void {
    const e = table[word];
    out.na = 0;
    out.ni = 0;
    var turn: u1 = e.start;
    var k: usize = 0;
    while (k < e.n) : (k += 1) {
        const len: u32 = e.lens[k];
        if (turn == 1) {
            out.active[out.na] = len;
            out.na += 1;
        } else {
            out.inactive[out.ni] = len;
            out.ni += 1;
        }
        turn +%= 1;
    }
}

test "p3 vs oracle: exhaustive u16" {
    try common.testExhaustive(count, runs);
}

/// Побитовый вариант: раны берутся из LUT, но нагрузка честная —
/// по одному вызову на каждый бит. Колбеки получают ПОЗИЦИЮ
/// (см. base.iterateWord: сумма позиций несворачиваема и сверяема).
/// Ветка на тип — раз на ран, внутри рана только счётчик.
pub fn iterateWord(
    byte_base: u32,
    word: W,
    comptime on_inactive: ?fn (u32, u3) void,
    comptime on_active: ?fn (u32) void,
) void {
    const e = table[word];
    var pos: u32 = 0;
    var is_active = e.start == 1;
    var k: usize = 0;
    while (k < e.n) : (k += 1) {
        const len: u32 = e.lens[k];
        if (is_active) {
            if (on_active) |f| {
                var j: u32 = 0;
                while (j < len) : (j += 1) {
                    f(pos + j);
                }
            }
        } else {
            if (on_inactive) |f| {
                var j: u32 = 0;
                while (j < len) : (j += 1) {
                    const p = pos + j;
                    f(byte_base + (p >> 3), @truncate(p));
                }
            }
        }
        pos += len;
        is_active = !is_active;
    }
}

// ABI-стабильная точка для asm-дампов (шаг asm-iter) и внешних замеров.
export fn p3_countU16(w: u16) u64 {
    var s = common.Sums{};
    count(w, &s);
    return (s.active << 32) | s.inactive;
}

var g_pos: u32 = 0;
var g_base: u32 = 0;
var g_exp: [16]u1 = [_]u1{0} ** 16;

fn tBitA(pos: u32) void {
    // Позиция пословная 0..15 (глобальность — через byte_base у inactive).
    std.debug.assert(pos == g_pos);
    std.debug.assert(g_exp[g_pos] == 1);
    g_pos += 1;
}

fn tBitI(byte_id: u32, bit: u3) void {
    std.debug.assert(byte_id == g_base + (g_pos >> 3));
    std.debug.assert(bit == (g_pos & 7));
    std.debug.assert(g_exp[g_pos] == 0);
    g_pos += 1;
}

test "p3 iterateWord: exhaustive u16, one call per bit" {
    var w: u32 = 0;
    while (w < 65536) : (w += 1) {
        var b: u32 = 0;
        while (b < 16) : (b += 1) {
            g_exp[b] = @truncate((w >> @truncate(b)) & 1);
        }
        g_pos = 0;
        g_base = w * 2;
        iterateWord(w * 2, @truncate(w), tBitI, tBitA);
        try std.testing.expectEqual(@as(u32, 16), g_pos);
    }
}

test "p3 table invariants" {
    // Каждый entry покрывает ровно 16 бит, суммы сходятся с длинами.
    var v: u32 = 0;
    while (v < common.N) : (v += 1) {
        const e = table[@as(usize, v)];
        var total: u32 = 0;
        var sa: u32 = 0;
        var si: u32 = 0;
        var turn: u1 = e.start;
        var k: usize = 0;
        while (k < e.n) : (k += 1) {
            const len: u32 = e.lens[k];
            total += len;
            if (turn == 1) {
                sa += len;
            } else {
                si += len;
            }
            turn +%= 1;
        }
        try std.testing.expectEqual(@as(u32, 16), total);
        try std.testing.expectEqual(@as(u32, e.sum_a), sa);
        try std.testing.expectEqual(@as(u32, e.sum_i), si);
    }
}
