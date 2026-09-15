/// Speed comparison: flat linear word scan vs slice iterator chain.
/// Tree leg `new` = slice-powered iterateTargetBits streaming ids straight
/// into the summing callback (fair: same N per-bit adds as flat, hierarchy
/// overhead only for slice discovery). Tree leg `arith` = sumActiveIds with
/// O(slices) arithmetic, no per-bit loop (slice ceiling for dense data).
/// Old baseline (ReleaseFast, same machine) for reference:
///   sparse1M(1004): linear 0.010ms tree-old 0.024ms | dense100k: linear 0.127 tree-old 0.051
///   every3rd200k: linear 0.104 tree-old 0.161 | ultra10M(100): linear 0.043 tree-old 0.004
/// Array-generator generation (fill + sum over out, now deleted) for reference:
///   sparse1M-inactive 4.789ms -> 0.721ms after sort removal.
/// New legs: flat = single-pass ctz sum (== old linear); new = iterateTargetBits
/// streaming straight into the summing callback (no arrays, no alloc).
/// zig build bench -Doptimize=ReleaseFast
const std = @import("std");
const bit_tree = @import("bit_tree.zig");

const Allocator = std.mem.Allocator;
const FlatBitSet = bit_tree.FlatBitSet;
const BitState = bit_tree.BitState;
const BitTree = bit_tree.BitTree;

const Stopwatch = struct {
    io: std.Io,
    t0: std.Io.Timestamp,
    fn start(io: std.Io) @This() {
        return .{ .io = io, .t0 = std.Io.Timestamp.now(io, .awake) };
    }
    fn read(self: *@This()) u64 {
        const t1 = std.Io.Timestamp.now(self.io, .awake);
        return @intCast(std.Io.Timestamp.durationTo(self.t0, t1).nanoseconds);
    }
};

fn fillStride(bits: anytype, stride: u32, offset: u32) void {
    var b: u32 = offset;
    while (b < bits.totalBitsCount()) : (b += stride) {
        bits.set(b, .active);
    }
}

fn fillClustered(bits: anytype, run: u32, gap: u32, offset: u32) void {
    const total: u64 = bits.totalBitsCount();
    var b: u64 = offset;
    while (b < total) {
        const end: u64 = @min(b + run, total);
        var i: u32 = @intCast(b);
        while (i < end) : (i += 1) {
            bits.set(i, .active);
        }
        b = end + gap;
    }
}

/// Previous algorithm: linear word scan, ctz per set word, sum on the fly, no store.
fn benchFlatSum(io: std.Io, flat: *const FlatBitSet, want: BitState, reps: u32) struct { sum: u64, count: u64, ns: u64 } {
    var watch = Stopwatch.start(io);
    var sum: u64 = 0;
    var count: u64 = 0;
    var r: u32 = 0;
    while (r < reps) : (r += 1) {
        const words = flat.words.items;
        var w: usize = 0;
        while (w < words.len) : (w += 1) {
            var bits: u64 = words[w];
            if (want == .inactive) bits = ~bits;
            if (w + 1 == words.len and (flat.total_bits & 63) != 0) {
                const rem: u6 = @intCast(flat.total_bits & 63);
                bits &= (@as(u64, 1) << rem) - 1;
            }
            while (bits != 0) {
                const s: u32 = @ctz(bits);
                bits &= bits - 1;
                sum += @as(u64, w) * 64 + s;
                count += 1;
            }
        }
    }
    return .{ .sum = sum, .count = count, .ns = watch.read() };
}

/// Summing callback context for the tree leg: work happens inside the walk.
const SumCtx = struct {
    sum: u64 = 0,
    count: u64 = 0,
    fn add(self: *SumCtx, id: u32) bool {
        self.sum += id;
        self.count += 1;
        return true;
    }
};

/// New algorithm: comptime iterator chain streams ids straight into `add`.
/// No output arrays, no frontier allocation, callbacks fully inlined.
fn benchTreeForEach(io: std.Io, tree: *const BitTree, comptime want: BitState, reps: u32) struct { sum: u64, count: u64, ns: u64 } {
    var watch = Stopwatch.start(io);
    var sum: u64 = 0;
    var count: u64 = 0;
    var r: u32 = 0;
    while (r < reps) : (r += 1) {
        var c = SumCtx{};
        const done = if (want == .active)
            tree.iterateTargetBits(*SumCtx, &c, SumCtx.add, null)
        else
            tree.iterateTargetBits(*SumCtx, &c, null, SumCtx.add);
        std.mem.doNotOptimizeAway(done);
        sum += c.sum;
        count += c.count;
    }
    return .{ .sum = sum, .count = count, .ns = watch.read() };
}

/// Ceiling leg: O(slices) arithmetic sum via sumActiveIds, no per-bit loop.
/// Same total as the fair legs; measures slice discovery + formula cost only.
fn benchTreeArith(io: std.Io, tree: *const BitTree, reps: u32) struct { sum: u64, count: u64, ns: u64 } {
    var watch = Stopwatch.start(io);
    var sum: u64 = 0;
    var count: u64 = 0;
    var r: u32 = 0;
    while (r < reps) : (r += 1) {
        const got = tree.sumActiveIds();
        std.mem.doNotOptimizeAway(got);
        sum += got.sum;
        count += got.count;
    }
    return .{ .sum = sum, .count = count, .ns = watch.read() };
}

/// One active-scan scenario with an extra arith row (fair + ceiling side by side).
fn benchArithPair(io: std.Io, alloc: Allocator, name: []const u8, tree: *const BitTree, flat: *const FlatBitSet, reps: u32) !void {
    _ = alloc;
    {
        const got = tree.sumActiveIds();
        std.mem.doNotOptimizeAway(got);
    }
    const fl = benchFlatSum(io, flat, .active, reps);
    const tr = benchTreeArith(io, tree, reps);
    std.debug.assert(fl.sum == tr.sum and fl.count == tr.count);
    printRow(name, fl.count / reps, @as(f64, @floatFromInt(fl.ns)) / @as(f64, @floatFromInt(reps)) / 1_000_000.0, @as(f64, @floatFromInt(tr.ns)) / @as(f64, @floatFromInt(reps)) / 1_000_000.0);
}
fn printHeader() void {
    std.debug.print("{s:<26} {s:>10} {s:>10} {s:>10} {s:>8}\n", .{ "pattern", "elements", "flat(ms)", "new(ms)", "xFlat" });
    std.debug.print("{s:<26} {s:>10} {s:>10} {s:>10} {s:>8}\n", .{ "--------------------------", "----------", "----------", "----------", "--------" });
}

fn printRow(name: []const u8, elements: u64, ms_flat: f64, ms_new: f64) void {
    const x: f64 = if (ms_new > 0) ms_flat / ms_new else 0;
    std.debug.print("{s:<26} {d:>10} {d:>10.3} {d:>10.3} {d:>7.2}x\n", .{ name, elements, ms_flat, ms_new, x });
}

/// One scenario on identical flat/tree contents. No output arrays anywhere.
fn benchPair(io: std.Io, alloc: Allocator, name: []const u8, tree: *const BitTree, flat: *const FlatBitSet, comptime want: BitState, reps: u32) !void {
    _ = alloc;
    // Warmup once (page in memory, branch predictors) outside the clock.
    {
        var c = SumCtx{};
        const done = if (want == .active)
            tree.iterateTargetBits(*SumCtx, &c, SumCtx.add, null)
        else
            tree.iterateTargetBits(*SumCtx, &c, null, SumCtx.add);
        std.mem.doNotOptimizeAway(done);
    }
    const fl = benchFlatSum(io, flat, want, reps);
    const tr = benchTreeForEach(io, tree, want, reps);
    std.debug.assert(fl.sum == tr.sum and fl.count == tr.count);
    printRow(name, fl.count / reps, @as(f64, @floatFromInt(fl.ns)) / @as(f64, @floatFromInt(reps)) / 1_000_000.0, @as(f64, @floatFromInt(tr.ns)) / @as(f64, @floatFromInt(reps)) / 1_000_000.0);
}

fn benchThresholdPair(io: std.Io, alloc: Allocator, bits_total: u32, stride: u32, reps: u32) !void {
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    var flat = FlatBitSet.empty;
    defer flat.deinit(alloc);
    try tree.resize(alloc, bits_total, .inactive);
    try flat.resize(alloc, bits_total, .inactive);
    fillStride(&tree, stride, 0);
    fillStride(&flat, stride, 0);
    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "gap{d}-N{d}", .{ stride, bits_total });
    try benchPair(io, alloc, name, &tree, &flat, .active, reps);
}

/// Write path: fresh tree + per-bit `set` calls. Exercises propagate.
/// Reports average fill time and per-set cost.
fn benchFill(io: std.Io, alloc: Allocator, bits_total: u32, stride: u32, offset: u32, reps: u32) !void {
    var watch = Stopwatch.start(io);
    var r: u32 = 0;
    var check: u64 = 0;
    while (r < reps) : (r += 1) {
        var tree = BitTree.empty;
        try tree.resize(alloc, bits_total, .inactive);
        var b: u32 = offset;
        while (b < bits_total) : (b += stride) {
            tree.set(b, .active);
        }
        check += tree.count(.active);
        tree.deinit(alloc);
    }
    std.mem.doNotOptimizeAway(check);
    const ns: f64 = @as(f64, @floatFromInt(watch.read())) / @as(f64, @floatFromInt(reps));
    const per_set: f64 = ns / @as(f64, @floatFromInt(check / reps));
    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "fill-s{d}-N{d}", .{ stride, bits_total });
    std.debug.print("{s:<26} {d:>10} {d:>10.3} {s:>10} {d:>7.1}ns/set\n", .{ name, check / reps, ns / 1_000_000.0, "---", per_set });
}

fn benchClusterPair(io: std.Io, alloc: Allocator, bits_total: u32, run: u32, gap: u32, reps: u32) !void {    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    var flat = FlatBitSet.empty;
    defer flat.deinit(alloc);
    try tree.resize(alloc, bits_total, .inactive);
    try flat.resize(alloc, bits_total, .inactive);
    fillClustered(&tree, run, gap, 0);
    fillClustered(&flat, run, gap, 0);
    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "cluster-r{d}-g{d}-N{d}", .{ run, gap, bits_total });
    try benchPair(io, alloc, name, &tree, &flat, .active, reps);
    const aname = try std.fmt.bufPrint(&name_buf, "cluster-r{d}-g{d}-N{d}-arith", .{ run, gap, bits_total });
    try benchArithPair(io, alloc, aname, &tree, &flat, reps);
}

pub fn main(init: std.process.Init) !void {
    const io: std.Io = init.io;
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc: Allocator = gpa.allocator();

    var sparse = FlatBitSet.empty;
    defer sparse.deinit(alloc);
    try sparse.resize(alloc, 1_000_000, .inactive);
    fillStride(&sparse, 997, 1);

    var dense = FlatBitSet.empty;
    defer dense.deinit(alloc);
    try dense.resize(alloc, 100_000, .active);

    var strided = FlatBitSet.empty;
    defer strided.deinit(alloc);
    try strided.resize(alloc, 200_000, .inactive);
    fillStride(&strided, 3, 0);

    var ultra = FlatBitSet.empty;
    defer ultra.deinit(alloc);
    try ultra.resize(alloc, 10_000_000, .inactive);
    fillStride(&ultra, 100_003, 7);

    var sparse_tree = BitTree.empty;
    defer sparse_tree.deinit(alloc);
    try sparse_tree.resize(alloc, 1_000_000, .inactive);
    fillStride(&sparse_tree, 997, 1);

    var dense_tree = BitTree.empty;
    defer dense_tree.deinit(alloc);
    try dense_tree.resize(alloc, 100_000, .active);

    var strided_tree = BitTree.empty;
    defer strided_tree.deinit(alloc);
    try strided_tree.resize(alloc, 200_000, .inactive);
    fillStride(&strided_tree, 3, 0);

    var ultra_tree = BitTree.empty;
    defer ultra_tree.deinit(alloc);
    try ultra_tree.resize(alloc, 10_000_000, .inactive);
    fillStride(&ultra_tree, 100_003, 7);

    printHeader();
    try benchPair(io, alloc, "sparse1M", &sparse_tree, &sparse, .active, 500);
    try benchArithPair(io, alloc, "sparse1M-arith", &sparse_tree, &sparse, 500);
    try benchPair(io, alloc, "sparse1M-inactive", &sparse_tree, &sparse, .inactive, 5);
    try benchPair(io, alloc, "dense100k", &dense_tree, &dense, .active, 20);
    try benchArithPair(io, alloc, "dense100k-arith", &dense_tree, &dense, 20);
    try benchPair(io, alloc, "every3rd200k", &strided_tree, &strided, .active, 20);
    try benchArithPair(io, alloc, "every3rd200k-arith", &strided_tree, &strided, 20);
    try benchPair(io, alloc, "ultra10M", &ultra_tree, &ultra, .active, 200);
    try benchArithPair(io, alloc, "ultra10M-arith", &ultra_tree, &ultra, 200);

    const strides = [_]u32{ 1, 2, 4, 8, 16, 64, 256, 1024, 4096, 16384, 65536 };
    for ([_]u32{1_000_000}) |total| {
        for (strides) |stride| {
            try benchThresholdPair(io, alloc, total, stride, 100);
        }
    }
    // N=10M sweep runs fewer reps to keep runtime sane (per-rep ms comparable).
    for (strides) |stride| {
        try benchThresholdPair(io, alloc, 10_000_000, stride, 5);
    }

    try benchClusterPair(io, alloc, 2_000_000, 1000, 1000, 20);
    try benchClusterPair(io, alloc, 2_000_000, 1000, 10000, 20);
    try benchClusterPair(io, alloc, 500_000, 50, 50, 50);
    try benchClusterPair(io, alloc, 500_000, 50, 500, 50);

    try benchFill(io, alloc, 1_000_000, 1, 0, 3);
    try benchFill(io, alloc, 1_000_000, 997, 1, 5);
}
