//! Отдельный бенч p8: только iterateByte, старые стратегии не гоняем.
//! Нагрузка в колбеках — += 1 (счётчик вызовов), суммы сверяются:
//! seq обязан дать 524288/524288, rnd — стабильные числа попарно.
//! шаг: zig build bench-p8 -Doptimize=ReleaseFast
const std = @import("std");
const common = @import("common.zig");
const p8 = @import("p8.zig");

var seq_buf: [common.N]u16 = undefined;
var rnd_buf: [common.N]u16 = undefined;

var g_a: u64 = 0;
var g_i: u64 = 0;

fn onA(bit: u3) void {
    _ = bit;
    g_a += 1;
}

fn onI(byte_id: u32, bit: u3) void {
    _ = byte_id;
    _ = bit;
    g_i += 1;
}

fn sweep(data: []const u16) void {
    var k: u32 = 0;
    for (data) |w| {
        p8.iterateByte(k * 2, @truncate(w), onI, onA);
        p8.iterateByte(k * 2 + 1, @truncate(w >> 8), onI, onA);
        k += 1;
    }
}

pub fn main(init: std.process.Init) !void {
    const io: std.Io = init.io;
    common.fillSeq(&seq_buf);
    common.fillRnd(&rnd_buf, common.RND_SEED);

    const reps: u64 = 30;
    sweep(&seq_buf);
    sweep(&rnd_buf); // warmup
    g_a = 0;
    g_i = 0;

    var watch = common.Stopwatch.start(io);
    var r: u64 = 0;
    while (r < reps) : (r += 1) sweep(&seq_buf);
    std.debug.print("p8 seq: {d:.3} ms/sweep a={d} i={d}\n", .{ watch.readMs(reps), g_a / reps, g_i / reps });

    g_a = 0;
    g_i = 0;
    watch = common.Stopwatch.start(io);
    r = 0;
    while (r < reps) : (r += 1) sweep(&rnd_buf);
    std.debug.print("p8 rnd: {d:.3} ms/sweep a={d} i={d}\n", .{ watch.readMs(reps), g_a / reps, g_i / reps });
}
