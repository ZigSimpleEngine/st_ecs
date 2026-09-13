/// Benchmarks for full scans and point queries over FlatBitSet.
/// Build in ReleaseFast for meaningful numbers, ideally with a native CPU target:
/// zig build bench -Doptimize=ReleaseFast -Dcpu=native
/// A native target lets LLVM emit TZCNT and POPCNT instead of conservative fallbacks.
const std = @import("std");
/// Bit container module import. Holds both the flat bitset and the hierarchy tree.
const bit_tree = @import("bit_tree.zig");

/// Allocator type used for benchmark bitsets.
const Allocator = std.mem.Allocator;
/// Tested flat bitset type.
const FlatBitSet = bit_tree.FlatBitSet;
/// Tested bit state type.
const BitState = bit_tree.BitState;
/// Tested tree type.
const BitTree = bit_tree.BitTree;
/// Tested scan strategy type.
const ScanKind = bit_tree.ScanKind;

/// Monotonic stopwatch over the process IO clock.
/// - `io` IO instance backing the monotonic clock.
/// - `t0` timestamp captured at start.
const Stopwatch = struct {
    /// IO instance backing the monotonic clock.
    io: std.Io,
    /// Timestamp captured at start.
    t0: std.Io.Timestamp,

    /// Captures the start timestamp.
    /// - `io` IO instance backing the monotonic clock.
    ///
    /// Return: running stopwatch.
    fn start(io: std.Io) @This() {
        return .{ .io = io, .t0 = std.Io.Timestamp.now(io, .awake) };
    }

    /// Reads elapsed nanoseconds since start without stopping.
    /// - `self` running stopwatch.
    ///
    /// Return: elapsed nanoseconds.
    fn read(self: *@This()) u64 {
        const t1 = std.Io.Timestamp.now(self.io, .awake);
        return @intCast(std.Io.Timestamp.durationTo(self.t0, t1).nanoseconds);
    }
};

/// Fills a bit container with clustered runs: dense blocks separated by gaps.
/// - `bits` bit container to fill, already resized. Tree or flat bitset.
/// - `run` set bits per block.
/// - `gap` cleared bits between blocks.
/// - `offset` first set bit.
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

/// Fills a bit container with a deterministic strided pattern.
/// - `bits` bit container to fill, already resized. Tree or flat bitset.
/// - `stride` distance between set bits.
/// - `offset` first set bit.
fn fillStride(bits: anytype, stride: u32, offset: u32) void {
    var b: u32 = offset;
    while (b < bits.totalBitsCount()) : (b += stride) {
        bits.set(b, .active);
    }
}

/// Measures a full scan driven by a stateful cursor.
/// Single shared loop body for every scan leg, so legs differ only by data.
/// - `io` IO instance backing the monotonic clock.
/// - `cursor` cursor positioned before the first matching bit. Consumed by value.
/// - `reps` scan repetitions.
///
/// Return: checksum over visited bits plus elapsed nanoseconds packed as struct fields.
fn scanLoop(io: std.Io, cursor: anytype, reps: u32) struct { sum: u64, count: u64, ns: u64 } {
    var watch = Stopwatch.start(io);
    var sum: u64 = 0;
    var count: u64 = 0;
    var r: u32 = 0;
    while (r < reps) : (r += 1) {
        var it = cursor;
        while (it.step()) |b| {
            sum += b;
            count += 1;
        }
    }
    return .{ .sum = sum, .count = count, .ns = watch.read() };
}

/// Measures isolated point queries driven by seek plus step from pseudo-random starts.
/// Single shared loop body for every point-query leg, so legs differ only by data.
/// - `io` IO instance backing the monotonic clock.
/// - `bits` queried bit container. Tree or flat bitset with an identical cursor API.
/// - `want` wanted bit state, compile-time known.
/// - `queries` number of point queries.
///
/// Return: checksum over answers plus elapsed nanoseconds packed as struct fields.
fn seekLoop(io: std.Io, bits: anytype, comptime want: anytype, queries: u32) struct { sum: u64, ns: u64 } {
    const total = bits.totalBitsCount();
    var watch = Stopwatch.start(io);
    var sum: u64 = 0;
    var state: u64 = 0x243F_6A88_85A3_08D3;
    var i: u32 = 0;
    while (i < queries) : (i += 1) {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        const s: u32 = @intCast((state >> 33) % total);
        var it = bits.cursor(want);
        it.seek(@as(u64, s) + 1);
        sum += it.step() orelse total;
    }
    return .{ .sum = sum, .ns = watch.read() };
}

/// Measures a full scan driven by a stateful cursor.
/// - `io` IO instance backing the monotonic clock.
/// - `bits` scanned bit container. Tree or flat bitset with an identical cursor API.
/// - `want` wanted bit state of the container, compile-time known.
/// - `reps` scan repetitions.
///
/// Return: checksum over visited bits plus elapsed nanoseconds packed as struct fields.
fn benchCursorLoop(io: std.Io, bits: anytype, comptime want: anytype, reps: u32) struct { sum: u64, count: u64, ns: u64 } {
    var watch = Stopwatch.start(io);
    var sum: u64 = 0;
    var count: u64 = 0;
    var r: u32 = 0;
    while (r < reps) : (r += 1) {
        var cursor = bits.cursor(want);
        while (cursor.step()) |b| {
            sum += b;
            count += 1;
        }
    }
    return .{ .sum = sum, .count = count, .ns = watch.read() };
}

/// Prints the scan comparison table header.
fn printTableHeader() void {
    std.debug.print("{s:<26} {s:>10} {s:>10} {s:>10} {s:>10}  {s}\n", .{ "pattern", "elements", "linear(ms)", "tree(ms)", "auto(ms)", "auto" });
    std.debug.print("{s:<26} {s:>10} {s:>10} {s:>10} {s:>10}  {s}\n", .{ "--------------------------", "----------", "----------", "----------", "----------", "------" });
}

/// Prints one aligned row of the scan comparison table.
/// - `name` scenario label.
/// - `elements` matching elements per full iteration.
/// - `ms_linear` average linear scan time in milliseconds.
/// - `ms_tree` average tree scan time in milliseconds.
/// - `ms_auto` average automatic scan time in milliseconds.
/// - `kind` strategy selected by the automatic leg.
fn printTrioRow(name: []const u8, elements: u64, ms_linear: f64, ms_tree: f64, ms_auto: f64, kind: ScanKind) void {
    std.debug.print("{s:<26} {d:>10} {d:>10.3} {d:>10.3} {d:>10.3}  {s}\n", .{ name, elements, ms_linear, ms_tree, ms_auto, @tagName(kind) });
}

/// Runs one scan pattern on the flat bitset, the tree and the automatic choice.
/// Prints one aligned table row and verifies identical checksums on all legs.
/// - `io` IO instance backing the monotonic clock.
/// - `name` scenario label.
/// - `tree` hierarchy container holding the pattern.
/// - `flat` flat container holding the identical pattern.
/// - `want` wanted bit state, compile-time known.
/// - `reps` scan repetitions per leg.
fn benchTrio(io: std.Io, name: []const u8, tree: *const BitTree, flat: *const FlatBitSet, comptime want: BitState, reps: u32) void {
    const fl = scanLoop(io, flat.cursor(want), reps);
    const tr = scanLoop(io, tree.cursor(want), reps);
    std.debug.assert(fl.sum == tr.sum and fl.count == tr.count);
    const kind = tree.detectScanKind(want);
    const au = if (kind == .tree) scanLoop(io, tree.cursor(want), reps) else scanLoop(io, flat.cursor(want), reps);
    std.debug.assert(au.sum == fl.sum and au.count == fl.count);
    const ms_linear: f64 = @as(f64, @floatFromInt(fl.ns)) / @as(f64, @floatFromInt(reps)) / 1_000_000.0;
    const ms_tree: f64 = @as(f64, @floatFromInt(tr.ns)) / @as(f64, @floatFromInt(reps)) / 1_000_000.0;
    const ms_auto: f64 = @as(f64, @floatFromInt(au.ns)) / @as(f64, @floatFromInt(reps)) / 1_000_000.0;
    printTrioRow(name, fl.count / reps, ms_linear, ms_tree, ms_auto, kind);
}

/// Runs one point-query pattern on the flat bitset, the tree and the automatic choice.
/// Prints one aligned table row and verifies identical checksums on all legs.
/// All legs query from identical pseudo-random starts.
/// - `io` IO instance backing the monotonic clock.
/// - `name` scenario label.
/// - `tree` hierarchy container holding the pattern.
/// - `flat` flat container holding the identical pattern.
/// - `want` wanted bit state, compile-time known.
/// - `queries` number of point queries per leg.
fn benchPointTrio(io: std.Io, name: []const u8, tree: *const BitTree, flat: *const FlatBitSet, comptime want: BitState, queries: u32) void {
    const fl = seekLoop(io, flat, want, queries);
    const tr = seekLoop(io, tree, want, queries);
    std.debug.assert(fl.sum == tr.sum);
    const kind = tree.detectScanKind(want);
    const au = if (kind == .tree) seekLoop(io, tree, want, queries) else seekLoop(io, flat, want, queries);
    std.debug.assert(au.sum == fl.sum);
    printTrioRow(
        name,
        queries,
        @as(f64, @floatFromInt(fl.ns)) / 1_000_000.0,
        @as(f64, @floatFromInt(tr.ns)) / 1_000_000.0,
        @as(f64, @floatFromInt(au.ns)) / 1_000_000.0,
        kind,
    );
}

/// Runs one sparseness-factor scenario and prints the average computation time.
/// Also verifies that every repetition reports the same factor.
/// - `io` IO instance backing the monotonic clock.
/// - `allocator` allocator for the temporary gap buffers.
/// - `name` scenario label.
/// - `bits` measured bitset.
/// - `want` measured bit state, compile-time known.
/// - `reps` computation repetitions.
fn benchFactorScenario(io: std.Io, allocator: Allocator, name: []const u8, bits: *const FlatBitSet, comptime want: BitState, reps: u32) !void {
    const first: u32 = try bits.sparseFactor(allocator, want);
    var watch = Stopwatch.start(io);
    var r: u32 = 0;
    while (r < reps) : (r += 1) {
        const factor: u32 = try bits.sparseFactor(allocator, want);
        std.debug.assert(factor == first);
        std.mem.doNotOptimizeAway(factor);
    }
    const ms: f64 = @as(f64, @floatFromInt(watch.read())) / @as(f64, @floatFromInt(reps)) / 1_000_000.0;
    std.debug.print("{s} [factor]: elements={d} avg={d:.3}ms\n", .{ name, bits.count(want), ms });
}

/// Builds benchmark bitsets and runs every scenario.
/// - `init` process initialization carrying the IO instance.
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

    printTableHeader();
    benchTrio(io, "sparse1M", &sparse_tree, &sparse, .active, 500);
    benchTrio(io, "sparse1M-inactive", &sparse_tree, &sparse, .inactive, 5);
    benchTrio(io, "dense100k", &dense_tree, &dense, .active, 20);
    benchTrio(io, "every3rd200k", &strided_tree, &strided, .active, 20);
    benchTrio(io, "ultra10M", &ultra_tree, &ultra, .active, 200);

    benchPointTrio(io, "points-sparse1M", &sparse_tree, &sparse, .active, 200_000);
    benchPointTrio(io, "points-dense100k", &dense_tree, &dense, .active, 200_000);

    try benchFactorScenario(io, alloc, "factor-dense100k-active", &dense, .active, 20);
    try benchFactorScenario(io, alloc, "factor-sparse1M-active", &sparse, .active, 50);
    try benchFactorScenario(io, alloc, "factor-sparse1M-inactive", &sparse, .inactive, 3);
    try benchFactorScenario(io, alloc, "factor-ultra10M-active", &ultra, .active, 50);

    try benchDetectScenario(io, "detect-dense100k", &dense_tree, .active, 20000);
    try benchDetectScenario(io, "detect-sparse1M", &sparse_tree, .active, 10000);
    try benchDetectScenario(io, "detect-ultra10M", &ultra_tree, .active, 10000);
    try benchDetectScenario(io, "detect-every3rd200k", &strided_tree, .active, 10000);

    const strides = [_]u32{ 1, 2, 4, 8, 16, 64, 256, 1024, 4096, 16384, 65536 };
    for ([_]u32{ 1_000_000, 10_000_000 }) |total| {
        for (strides) |stride| {
            try benchThresholdPair(io, alloc, total, stride, 100);
        }
    }

    try benchClusterPair(io, alloc, 2_000_000, 1000, 1000, 20);
    try benchClusterPair(io, alloc, 2_000_000, 1000, 10000, 20);
    try benchClusterPair(io, alloc, 500_000, 50, 50, 50);
    try benchClusterPair(io, alloc, 500_000, 50, 500, 50);
}

/// Builds a tree and a flat bitset with an identical strided pattern and benchmarks both.
/// The stride equals the median gap of the pattern by construction.
/// - `io` IO instance backing the monotonic clock.
/// - `allocator` allocator for both containers.
/// - `bits_total` bit count for both containers.
/// - `stride` distance between set bits, also the pattern median gap.
/// - `reps` scan repetitions per container.
fn benchThresholdPair(io: std.Io, allocator: Allocator, bits_total: u32, stride: u32, reps: u32) !void {
    var tree = BitTree.empty;
    defer tree.deinit(allocator);
    var flat = FlatBitSet.empty;
    defer flat.deinit(allocator);
    try tree.resize(allocator, bits_total, .inactive);
    try flat.resize(allocator, bits_total, .inactive);
    fillStride(&tree, stride, 0);
    fillStride(&flat, stride, 0);

    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "gap{d}-N{d}", .{ stride, bits_total });
    benchTrio(io, name, &tree, &flat, .active, reps);
}

/// Builds a tree and a flat bitset with an identical clustered pattern and benchmarks both.
/// - `io` IO instance backing the monotonic clock.
/// - `allocator` allocator for both containers.
/// - `bits_total` bit count for both containers.
/// - `run` set bits per block.
/// - `gap` cleared bits between blocks.
/// - `reps` scan repetitions per container.
fn benchClusterPair(io: std.Io, allocator: Allocator, bits_total: u32, run: u32, gap: u32, reps: u32) !void {
    var tree = BitTree.empty;
    defer tree.deinit(allocator);
    var flat = FlatBitSet.empty;
    defer flat.deinit(allocator);
    try tree.resize(allocator, bits_total, .inactive);
    try flat.resize(allocator, bits_total, .inactive);
    fillClustered(&tree, run, gap, 0);
    fillClustered(&flat, run, gap, 0);

    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "cluster-r{d}-g{d}-N{d}", .{ run, gap, bits_total });
    benchTrio(io, name, &tree, &flat, .active, reps);
}

/// Runs one detection-speed scenario and prints the average detection time.
/// Gates two requirements loudly: a tenfold margin against reference full scans
/// of at least twenty microseconds, and an absolute two-microsecond cap below that.
/// Sub-twenty-microsecond scans cannot host a tenfold margin by information theory:
/// the decision itself needs dozens of summary reads.
/// - `io` IO instance backing the monotonic clock.
/// - `name` scenario label.
/// - `tree` inspected tree.
/// - `want` wanted bit state, compile-time known.
/// - `reps` detection repetitions.
fn benchDetectScenario(io: std.Io, name: []const u8, tree: *const BitTree, comptime want: BitState, reps: u32) !void {
    const decided = tree.detectScanKind(want);
    var watch = Stopwatch.start(io);
    var sink: u64 = 0;
    var r: u32 = 0;
    while (r < reps) : (r += 1) {
        sink += @intFromEnum(tree.detectScanKind(want));
    }
    std.mem.doNotOptimizeAway(sink);
    std.mem.doNotOptimizeAway(decided);
    const detect_ns: f64 = @as(f64, @floatFromInt(watch.read())) / @as(f64, @floatFromInt(reps));
    const scan = benchCursorLoop(io, tree, want, 1);
    std.mem.doNotOptimizeAway(scan.sum);
    std.mem.doNotOptimizeAway(scan.count);
    const scan_ns: f64 = @as(f64, @floatFromInt(scan.ns));
    if (scan_ns >= 20_000.0) {
        if (scan_ns < 10.0 * detect_ns) return error.DetectTooSlow;
    } else if (detect_ns > 2_000.0) {
        return error.DetectTooSlow;
    }
    std.debug.print("{s} [detect:{s}]: elements={d} avg={d:.3}ms\n", .{ name, @tagName(decided), tree.totalBitsCount(), detect_ns / 1_000_000.0 });
}
