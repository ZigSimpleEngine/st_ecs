//! BASE: текущая стратегия — active_flag, ветка на тип каждого рана.
//! Контрольная точка для сравнения прототипов.
const std = @import("std");
const common = @import("common.zig");
const W = common.W;
const bw = common.bw;
const BITS = common.BITS;

pub fn count(word: W, sums: *common.Sums) void {
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

        if (bit_counter >= BITS) {
            if (active_flag == 0) {
                sums.inactive += BITS - old_bit_counter;
            } else {
                sums.active += BITS - old_bit_counter;
            }
            break;
        } else {
            if (active_flag == 0) {
                sums.inactive += ctz;
            } else {
                sums.active += ctz;
            }
        }

        active_flag +%= 1;
        word_mut = ~word_mut;
        word_mut >>= @truncate(ctz);
        word_mut &= bw.maskEndInverted(@truncate(bit_counter));
    }
}

pub fn runs(word: W, out: *common.Runs) void {
    const start_bit_value: u1 = @truncate(word & 1);
    var word_mut = if (start_bit_value == 0) word else ~word;
    var old_bit_counter: u32 = 0;
    var bit_counter: u32 = 0;
    var ctz: u32 = 0;
    var active_flag = start_bit_value;
    out.na = 0;
    out.ni = 0;

    while (true) {
        ctz = @ctz(word_mut);
        old_bit_counter = bit_counter;
        bit_counter += ctz;

        if (bit_counter >= BITS) {
            if (active_flag == 0) {
                out.inactive[out.ni] = BITS - old_bit_counter;
                out.ni += 1;
            } else {
                out.active[out.na] = BITS - old_bit_counter;
                out.na += 1;
            }
            break;
        } else {
            if (active_flag == 0) {
                out.inactive[out.ni] = ctz;
                out.ni += 1;
            } else {
                out.active[out.na] = ctz;
                out.na += 1;
            }
        }

        active_flag +%= 1;
        word_mut = ~word_mut;
        word_mut >>= @truncate(ctz);
        word_mut &= bw.maskEndInverted(@truncate(bit_counter));
    }
}

test "base vs oracle: exhaustive u16" {
    try common.testExhaustive(count, runs);
}

/// Побитовый вариант: тот же ctz-цикл поиска ранов, но нагрузка честная —
/// по одному вызову на каждый бит. Колбеки получают ПОЗИЦИЮ бита:
/// on_active(pos_in_word 0..15), on_inactive(byte_id, bit_in_byte).
/// Сумма позиций несворачиваема (значения зависят от данных), но сверяема
/// в замкнутой форме: полный перебор u16 даёт 32768 * (0+..+15) = 3932160
/// на каждую сторону. Ветка на тип — раз на ран.
pub fn iterateWord(
    byte_base: u32,
    word: W,
    comptime on_inactive: ?fn (u32, u3) void,
    comptime on_active: ?fn (u32) void,
) void {
    const start_bit_value: u1 = @truncate(word & 1);
    var word_mut = if (start_bit_value == 0) word else ~word;
    var bit_counter: u32 = 0;
    var active_flag = start_bit_value;

    while (true) {
        const ctz: u32 = @ctz(word_mut);
        const old = bit_counter;
        bit_counter += ctz;
        const len: u32 = if (bit_counter >= BITS) BITS - old else ctz;
        const done = bit_counter >= BITS;

        if (active_flag == 0) {
            if (on_inactive) |f| {
                var k: u32 = 0;
                while (k < len) : (k += 1) {
                    const p = old + k;
                    f(byte_base + (p >> 3), @truncate(p));
                }
            }
        } else {
            if (on_active) |f| {
                var k: u32 = 0;
                while (k < len) : (k += 1) {
                    f(old + k);
                }
            }
        }
        if (done) break;

        active_flag +%= 1;
        word_mut = ~word_mut;
        word_mut >>= @truncate(ctz);
        word_mut &= bw.maskEndInverted(@truncate(bit_counter));
    }
}

// ABI-стабильная точка для asm-дампов (шаг asm-iter) и внешних замеров.
export fn base_countU16(w: u16) u64 {
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

test "base iterateWord: exhaustive u16, one call per bit" {
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
