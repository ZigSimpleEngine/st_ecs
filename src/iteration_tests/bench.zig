//! Символическая нагрузка: сумма ПОЗИЦИЙ бит (sum += pos), 16 вызовов
//! на слово для всех стратегий.
//! Несворачиваемость: значения зависят от данных, общей инструкции нет
//! (доказано экспериментом p8A-nofold: 0.63 мс честной работы).
//! Сверяемость: seq даёт 32768 * (0+..+15) = 3932160 на сторону
//! (замкнутая форма); rnd сверяется с одноразовым reference-сканом.
//! Позиции пословные 0..15 (byte_base = 0 везде): base/p3 отдают pos
//! напрямую, p8/p9 — bit и bit+8 для lo/hi байтов. Суммы сравнимы.
//! Детектор свёртки: exact-суммы + вменяемое время (>= 0.3 мс — честная
//! работа; ~0.05 мс и ниже = popcount-свёртка, как было с p9A/p9A64).
const std = @import("std");
const common = @import("common.zig");
const base = @import("base.zig");
const p3 = @import("p3.zig");
const p8 = @import("p8.zig");
const p9 = @import("p9.zig");

var seq_buf: [common.N]u16 = undefined;
var rnd_buf: [common.N]u16 = undefined;

var g_pa: u64 = 0;
var g_pi: u64 = 0;

// Позиции пословные: active 0..15 напрямую (base/p3).
fn pA(pos: u32) void {
    g_pa += pos;
}

// Inactive: (byte_id, bit), byte_id = 0/1 при byte_base = 0.
fn pI(byte_id: u32, bit: u3) void {
    g_pi += @as(u64, byte_id) * 8 + bit;
}

// p8/p9: побайтовые колбеки, позиция собирается из bit (+8 hi).
fn pAlo(bit: u3) void {
    g_pa += bit;
}

fn pAhi(bit: u3) void {
    g_pa += @as(u64, bit) + 8;
}

fn pIlo(bit: u3) void {
    g_pi += bit;
}

fn pIhi(bit: u3) void {
    g_pi += @as(u64, bit) + 8;
}

fn wBase(_: u32, w: u16) void {
    base.iterateWord(0, w, pI, pA);
}

fn wP3(_: u32, w: u16) void {
    p3.iterateWord(0, w, pI, pA);
}

fn wP8(_: u32, w: u16) void {
    p8.iterateByte(0, @truncate(w), pI, pAlo);
    p8.iterateByte(1, @truncate(w >> 8), pI, pAhi);
}

fn wBaseA(_: u32, w: u16) void {
    base.iterateWord(0, w, null, pA);
}

fn wP3A(_: u32, w: u16) void {
    p3.iterateWord(0, w, null, pA);
}

fn wP8A(_: u32, w: u16) void {
    p8.iterateByte(0, @truncate(w), null, pAlo);
    p8.iterateByte(1, @truncate(w >> 8), null, pAhi);
}

fn wP9A(_: u32, w: u16) void {
    p9.iterateActiveByte(@truncate(w), pAlo);
    p9.iterateActiveByte(@truncate(w >> 8), pAhi);
}

fn wP9I(_: u32, w: u16) void {
    p9.iterateInactiveByte(@truncate(w), pIlo);
    p9.iterateInactiveByte(@truncate(w >> 8), pIhi);
}

fn bench(
    name: []const u8,
    data: []const u16,
    comptime f: fn (u32, u16) void,
    io: std.Io,
    exp_a: u64,
    exp_i: u64,
) !void {
    for (data) |w| f(0, w); // warmup
    g_pa = 0;
    g_pi = 0;
    const reps: u64 = 30;
    var watch = common.Stopwatch.start(io);
    var r: u64 = 0;
    while (r < reps) : (r += 1) {
        for (data) |w| f(0, w);
    }
    const ms = watch.readMs(reps);
    const a = g_pa / reps;
    const i = g_pi / reps;
    std.debug.print("{s}: {d:.3} ms/sweep a={d} i={d}\n", .{ name, ms, a, i });
    if (a != exp_a or i != exp_i) {
        std.debug.print("  MISMATCH! expect a={d} i={d}\n", .{ exp_a, exp_i });
        return error.TestExpectedEqual;
    }
}

pub fn main(init: std.process.Init) !void {
    const io: std.Io = init.io;
    common.fillSeq(&seq_buf);
    common.fillRnd(&rnd_buf, common.RND_SEED);

    // Замкнутая форма seq: каждая позиция 0..15 стоит в 32768 словах.
    const seq_a: u64 = 32768 * 120;
    const seq_i: u64 = 32768 * 120;

    // Одноразовый reference-скан rnd (вне замера).
    var exp_ra: u64 = 0;
    var exp_ri: u64 = 0;
    for (rnd_buf) |w| {
        var b: u32 = 0;
        while (b < 16) : (b += 1) {
            if (((w >> @truncate(b)) & 1) == 1) {
                exp_ra += b;
            } else {
                exp_ri += b;
            }
        }
    }

    std.debug.print("--- seq ---\n", .{});
    try bench("base", &seq_buf, wBase, io, seq_a, seq_i);
    try bench("p3  ", &seq_buf, wP3, io, seq_a, seq_i);
    try bench("p8  ", &seq_buf, wP8, io, seq_a, seq_i);
    try bench("baseA", &seq_buf, wBaseA, io, seq_a, 0);
    try bench("p3A  ", &seq_buf, wP3A, io, seq_a, 0);
    try bench("p8A  ", &seq_buf, wP8A, io, seq_a, 0);
    try bench("p9A  ", &seq_buf, wP9A, io, seq_a, 0);
    try bench("p9I  ", &seq_buf, wP9I, io, 0, seq_i);

    std.debug.print("--- rnd ---\n", .{});
    try bench("base", &rnd_buf, wBase, io, exp_ra, exp_ri);
    try bench("p3  ", &rnd_buf, wP3, io, exp_ra, exp_ri);
    try bench("p8  ", &rnd_buf, wP8, io, exp_ra, exp_ri);
    try bench("baseA", &rnd_buf, wBaseA, io, exp_ra, 0);
    try bench("p3A  ", &rnd_buf, wP3A, io, exp_ra, 0);
    try bench("p8A  ", &rnd_buf, wP8A, io, exp_ra, 0);
    try bench("p9A  ", &rnd_buf, wP9A, io, exp_ra, 0);
    try bench("p9I  ", &rnd_buf, wP9I, io, 0, exp_ri);
}
