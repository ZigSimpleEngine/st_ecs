//! Общий код стратегий итерации по ранам слова.
//! Идея всех стратегий: последовательные длины active/inactive ранов слова.
const std = @import("std");
const BitWord = @import("bit_word").BitWord;

pub const W = u16;
pub const bw = BitWord(W);
pub const BITS: u32 = bw.word_type_bits;
pub const MAX_RUNS: usize = 16;
pub const N: usize = 65536;
pub const RND_SEED: u64 = 0x9E3779B97F4A7C15;

pub const Sums = struct {
    active: u64 = 0,
    inactive: u64 = 0,
};

pub const Runs = struct {
    active: [MAX_RUNS]u32 = undefined,
    na: usize = 0,
    inactive: [MAX_RUNS]u32 = undefined,
    ni: usize = 0,
};

/// Побитовый эталон: бежит по битам, режет раны. Не зависит от стратегий.
pub fn oracleRuns(word: W, out: *Runs) void {
    var cur: u1 = @truncate(word & 1);
    var len: u32 = 0;
    out.na = 0;
    out.ni = 0;
    var b: u32 = 0;
    while (b < BITS) : (b += 1) {
        const bit: u1 = @truncate((word >> @truncate(b)) & 1);
        if (bit == cur) {
            len += 1;
        } else {
            if (cur == 1) {
                out.active[out.na] = len;
                out.na += 1;
            } else {
                out.inactive[out.ni] = len;
                out.ni += 1;
            }
            cur = bit;
            len = 1;
        }
    }
    if (cur == 1) {
        out.active[out.na] = len;
        out.na += 1;
    } else {
        out.inactive[out.ni] = len;
        out.ni += 1;
    }
}

pub fn expectRunsEqual(exp: *const Runs, got: *const Runs) !void {
    try std.testing.expectEqualSlices(u32, exp.active[0..exp.na], got.active[0..got.na]);
    try std.testing.expectEqualSlices(u32, exp.inactive[0..exp.ni], got.inactive[0..got.ni]);
}

/// Полный перебор u16 против эталона + проверка симметрии сумм.
/// countFn и runsFn — функции конкретной стратегии.
pub fn testExhaustive(
    comptime countFn: fn (W, *Sums) void,
    comptime runsFn: fn (W, *Runs) void,
) !void {
    var exp: Runs = .{};
    var got: Runs = .{};
    var s = Sums{};
    var v: u32 = 0;
    while (v < N) : (v += 1) {
        const w: W = @truncate(v);
        oracleRuns(w, &exp);
        got = .{};
        runsFn(w, &got);
        try expectRunsEqual(&exp, &got);
        countFn(w, &s);
    }
    // Симметрия полного перебора: ровно половина битов active.
    try std.testing.expectEqual(@as(u64, N * BITS / 2), s.active);
    try std.testing.expectEqual(@as(u64, N * BITS / 2), s.inactive);
}

/// Детерминированный xorshift64* для набора случайных данных (фикс-сид).
pub const Rng = struct {
    s: u64,
    pub fn init(seed: u64) Rng {
        return .{ .s = seed };
    }
    pub fn next(r: *Rng) u64 {
        var x = r.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        r.s = x;
        return x *% 0x2545F4914F6CDD1D;
    }
    pub fn nextU16(r: *Rng) u16 {
        return @truncate(r.next() >> 48);
    }
};

pub fn fillSeq(buf: []W) void {
    var i: u32 = 0;
    while (i < buf.len) : (i += 1) buf[i] = @truncate(i);
}

pub fn fillRnd(buf: []W, seed: u64) void {
    var rng = Rng.init(seed);
    for (buf) |*p| p.* = rng.nextU16();
}

pub const Stopwatch = struct {
    io: std.Io,
    t0: std.Io.Timestamp,
    pub fn start(io: std.Io) @This() {
        return .{ .io = io, .t0 = std.Io.Timestamp.now(io, .awake) };
    }
    pub fn readMs(self: *@This(), reps: u64) f64 {
        const t1 = std.Io.Timestamp.now(self.io, .awake);
        const ns: u64 = @intCast(std.Io.Timestamp.durationTo(self.t0, t1).nanoseconds);
        return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(reps)) / 1.0e6;
    }
};

// ============ Обобщения на произвольную ширину слова (для p4) ============

/// Буфер ранов на R слотов (R >= ширина слова в битах).
pub fn RunsN(comptime R: usize) type {
    return struct {
        active: [R]u32 = undefined,
        na: usize = 0,
        inactive: [R]u32 = undefined,
        ni: usize = 0,
    };
}

/// Побитовый эталон для произвольного беззнакового слова.
pub fn oracleRunsW(comptime Word: type, comptime R: usize, word: Word, out: *RunsN(R)) void {
    const bits: u32 = @bitSizeOf(Word);
    var cur: u1 = @truncate(word & 1);
    var len: u32 = 0;
    out.na = 0;
    out.ni = 0;
    var b: u32 = 0;
    while (b < bits) : (b += 1) {
        const bit: u1 = @truncate((word >> @truncate(b)) & 1);
        if (bit == cur) {
            len += 1;
        } else {
            if (cur == 1) {
                out.active[out.na] = len;
                out.na += 1;
            } else {
                out.inactive[out.ni] = len;
                out.ni += 1;
            }
            cur = bit;
            len = 1;
        }
    }
    if (cur == 1) {
        out.active[out.na] = len;
        out.na += 1;
    } else {
        out.inactive[out.ni] = len;
        out.ni += 1;
    }
}

pub fn expectRunsNEqual(comptime R: usize, exp: *const RunsN(R), got: *const RunsN(R)) !void {
    try std.testing.expectEqualSlices(u32, exp.active[0..exp.na], got.active[0..got.na]);
    try std.testing.expectEqualSlices(u32, exp.inactive[0..exp.ni], got.inactive[0..got.ni]);
}

/// Одна проверка слова против эталона (общая для exhaustive/sample/edges).
/// При расхождении печатает слово и обе последовательности (диагностика).
pub fn checkWordW(
    comptime Word: type,
    comptime R: usize,
    word: Word,
    comptime runsFn: fn (Word, *RunsN(R)) void,
) !void {
    var exp: RunsN(R) = .{};
    var got: RunsN(R) = .{};
    oracleRunsW(Word, R, word, &exp);
    runsFn(word, &got);
    const ok_a = std.mem.eql(u32, exp.active[0..exp.na], got.active[0..got.na]);
    const ok_i = std.mem.eql(u32, exp.inactive[0..exp.ni], got.inactive[0..got.ni]);
    if (!ok_a or !ok_i) {
        std.debug.print("MISMATCH word=0x{x} ({}):\n  exp a={any} i={any}\n  got a={any} i={any}\n", .{
            word,  word,
            exp.active[0..exp.na],   exp.inactive[0..exp.ni],
            got.active[0..got.na],   got.inactive[0..got.ni],
        });
        return error.TestExpectedEqual;
    }
}

/// Полный перебор всего диапазона слова (реально только для u8/u16).
pub fn testExhaustiveW(
    comptime Word: type,
    comptime R: usize,
    comptime countFn: fn (Word, *Sums) void,
    comptime runsFn: fn (Word, *RunsN(R)) void,
) !void {
    const total: u64 = @as(u64, 1) << @bitSizeOf(Word);
    var s = Sums{};
    var v: u64 = 0;
    while (v < total) : (v += 1) {
        const w: Word = @truncate(v);
        try checkWordW(Word, R, w, runsFn);
        countFn(w, &s);
    }
    try std.testing.expectEqual(total * @bitSizeOf(Word) / 2, s.active);
    try std.testing.expectEqual(total * @bitSizeOf(Word) / 2, s.inactive);
}

/// Случайная выборка фиксированным сидом: ловит пересечения границ байтов.
pub fn testSampleW(
    comptime Word: type,
    comptime R: usize,
    comptime runsFn: fn (Word, *RunsN(R)) void,
    comptime n: usize,
    comptime seed: u64,
) !void {
    var rng = Rng.init(seed);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const w: Word = @truncate(rng.next());
        try checkWordW(Word, R, w, runsFn);
    }
}
