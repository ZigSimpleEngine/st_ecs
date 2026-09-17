const std = @import("std");
const math = std.math;
const utilities = @import("utilities.zig");
const bit_word = @import("bit_word.zig");
const BitSet = @import("bit_set.zig").BitSet;
const Layer = @import("layer.zig").Layer;

const IteratorCallback = utilities.IteratorCallback;
const Allocator = std.mem.Allocator;
const ListA64 = utilities.ListA64;
const BitState = utilities.BitState;

pub fn BitTree(Word: type) type {
    comptime {
        utilities.assertUnsignedPowerOfTwoInt(Word);
    }

    const bw = bit_word.BitWord(Word);

    return struct {
        const Self = @This();

        bitset: BitSet(Word) = .{},
        layers: ListA64(Layer(Word)) = .empty,

        pub fn TreeWithContext(Context: type) type {
            return struct {
                tree: *Self,
                context: Context,
            };
        }

        pub fn Iterator(
            comptime Context: type,
            comptime on_active: IteratorCallback(Context),
            comptime on_inactive: IteratorCallback(Context),
        ) type {
            return struct {
                const L = Layer(Word);
                const B = BitSet(Word);

                const active_unwrapper = ActivityUnwrapper(on_active).unwrap;
                const inactive_unwrapper = ActivityUnwrapper(on_inactive).unwrap;

                const LI = L.Iterator(
                    *LayerWithContext,
                    inactive_unwrapper,
                    active_unwrapper,
                    mixed_unwrapper,
                    deep_mixed_unwrapper,
                );

                const LayerWithContext = TreeWithContext(struct {
                    layer_id: u32,
                    context: Context,
                });

                fn ActivityUnwrapper(comptime callback: IteratorCallback(Context)) type {
                    return struct {
                        inline fn unwrap(data: *LayerWithContext, word_id: u32) bool {
                            const layer_id = data.context.layer_id;
                            const context = data.context.context;
                            const end_word_id = word_id + 1;
                            const shift: bw.Shift = @truncate(bw.shift_type_bits * layer_id);
                            const start_bit_id = word_id << shift;
                            const end_bit_id = end_word_id << shift;

                            for (start_bit_id..end_bit_id) |i| {
                                if (callback) |f| {
                                    if (!f(context, @truncate(i))) return false;
                                }
                            }
                        }
                    };
                }

                fn mixed_unwrapper(data: *LayerWithContext, word_id: u32) bool {
                    data.context.layer_id -= 1;
                    const end_word_id = word_id + 1;
                    const start_bit_id = word_id << bw.shift_type_bits;
                    const end_bit_id = end_word_id << bw.shift_type_bits;

                    for (start_bit_id..end_bit_id) |i| {
                        LI.step(data, @truncate(i));
                    }
                }

                inline fn deep_mixed_unwrapper(data: *LayerWithContext, word_id: u32) bool {
                    const layer_id = data.context.layer_id;
                    const end_word_id = word_id + 1;
                    const shift: bw.Shift = @truncate(bw.shift_type_bits * (layer_id - 1));
                    const start_bit_id = word_id << shift;
                    const end_bit_id = end_word_id << shift;

                    const BI = B.Iterator(Context, on_active, on_inactive);
                    const bi_data: B.BitsetWithContext(Context) = .{
                        .bitset = data.tree.bitset,
                        .context = data.context.context,
                    };
                    for (start_bit_id..end_bit_id) |i| {
                        if (!BI.step(bi_data, @truncate(i))) return false;
                    }
                }

                pub inline fn step(data: TreeWithContext(Context)) bool {
                    const layers = data.tree.layers.items;
                    var context: LayerWithContext = .{
                        .context = .{
                            .context = data.context,
                            .layer_id = layers.len,
                        },
                        .tree = data.tree,
                    };

                    for (layers, 0..) |_, i| {
                        if (!LI.step(&context, i)) return false;
                    }

                    return true;
                }
            };
        }

        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.bitset.deinit(allocator);
            for (self.layers.items) |*layer| {
                layer.deinit(allocator);
            }
            self.layers.deinit(allocator);
        }

        pub fn setWord(self: *Self, id: u32, word: Word, mask: Word) void {
            self.bitset.setWord(id, word, mask);
            self.layers.items[0].setWord(id, word, 0, mask);
            var wid = id;
            var li: usize = 1;
            while (li < self.layers.items.len) : (li += 1) {
                self.refreshWord(li, wid);
                wid = bw.bitToWordId(wid);
            }
        }

        pub fn setBit(self: *Self, id: u32, value: BitState) void {
            self.bitset.setBit(id, value);
            const leaf: Layer(Word).State = if (value == .active) .active else .inactive;
            self.layers.items[0].setBit(id, leaf);
            var wid = bw.bitToWordId(id);
            var li: usize = 1;
            while (li < self.layers.items.len) : (li += 1) {
                self.refreshWord(li, wid);
                wid = bw.bitToWordId(wid);
            }
        }

        pub fn resize(self: *Self, allocator: Allocator, new_bits_count: u32, created_bits_value: BitState) !void {
            try self.bitset.resize(allocator, new_bits_count, created_bits_value);

            var want: usize = 0;
            if (new_bits_count > 0) {
                want = 1;
                var s = new_bits_count;
                while (s > 1) {
                    s = bw.bitsToWordsCount(s);
                    want += 1;
                }
            }
            while (self.layers.items.len > want) {
                const idx = self.layers.items.len - 1;
                var gone = self.layers.orderedRemove(idx);
                gone.deinit(allocator);
            }
            while (self.layers.items.len < want) {
                try self.layers.append(allocator, Layer(Word){});
            }

            const leaf_created: Layer(Word).State = if (created_bits_value == .active) .active else .inactive;
            var s = new_bits_count;
            for (self.layers.items, 0..) |*layer, li| {
                try layer.resize(allocator, s, if (li == 0) leaf_created else .inactive);
                s = bw.bitsToWordsCount(s);
            }
            var li: usize = 1;
            while (li < self.layers.items.len) : (li += 1) {
                const lower_words: u32 = @truncate(self.layers.items[li - 1].activity.items.len);
                var wid: u32 = 0;
                while (wid < lower_words) : (wid += 1) {
                    self.refreshWord(li, wid);
                }
            }
        }

        fn refreshWord(self: *Self, upper_layer: usize, word_id: u32) void {
            const L = Layer(Word);
            const lower = &self.layers.items[upper_layer - 1];
            const upper = &self.layers.items[upper_layer];
            const lower_used = bw.bitIdInWord(lower.bits_count);
            const last = lower.activity.items.len - 1;
            const valid: Word = if (word_id == last and lower_used != 0) bw.maskStart(lower_used) else bw.max_value;
            const st: L.State = if (upper_layer == 1) blk: {
                const lw: Word = lower.activity.items[word_id] & valid;
                break :blk if (lw == 0)
                    .inactive
                else if (lw == valid)
                    .active
                else
                    .deep_mixed;
            } else blk: {
                const aw: Word = lower.activity.items[word_id] & valid;
                const mw: Word = lower.mixed.items[word_id] & valid;
                const a_uni = aw == 0 or aw == valid;
                const m_uni = mw == 0 or mw == valid;
                break :blk if (a_uni and m_uni)
                    L.State.fromBits(
                        if (aw == 0) BitState.inactive else BitState.active,
                        if (mw == 0) BitState.inactive else BitState.active,
                    )
                else
                    .mixed;
            };
            upper.setBit(word_id, st);
        }
    };
}

const t = std.testing;

const TreeStepIds = struct {
    active: [512]u32 = undefined,
    na: usize = 0,
    inactive: [512]u32 = undefined,
    ni: usize = 0,
    stop_after: u32 = std.math.maxInt(u32),

    fn total(self: *const TreeStepIds) usize {
        return self.na + self.ni;
    }
};

fn treePushA(ctx: *TreeStepIds, bit_id: u32) bool {
    ctx.active[ctx.na] = bit_id;
    ctx.na += 1;
    return ctx.total() < ctx.stop_after;
}

fn treePushI(ctx: *TreeStepIds, bit_id: u32) bool {
    ctx.inactive[ctx.ni] = bit_id;
    ctx.ni += 1;
    return ctx.total() < ctx.stop_after;
}

fn treeStepCollectAll(comptime Word: type, tree: *BitTree(Word), ctx: *TreeStepIds) bool {
    const It = BitTree(Word).Iterator(*TreeStepIds, treePushA, treePushI);
    return It.step(.{ .tree = tree, .context = ctx });
}

/// Независимый эталон: побитовое чтение битсета 0..bits_count-1.
fn treeOracleIds(comptime Word: type, tree: *const BitTree(Word), target: BitState, out: *[512]u32) usize {
    const BW = bit_word.BitWord(Word);
    var n: usize = 0;
    var i: u32 = 0;
    while (i < tree.bitset.bits_count) : (i += 1) {
        const wid: usize = @intCast(BW.bitToWordId(i));
        if (BW.readBitState(tree.bitset.words.items[wid], BW.bitIdInWord(i)) == target) {
            out[n] = i;
            n += 1;
        }
    }
    return n;
}

/// Независимая сверка summary сверху вниз, бит за битом (не popcount):
/// слой 1 — 0/все/иначе в inactive/active/deep_mixed; выше — единогласие
/// детей в их стейт, иначе mixed.
fn expectSummariesValid(comptime Word: type, tree: *const BitTree(Word)) !void {
    const BW = bit_word.BitWord(Word);
    const L = Layer(Word);
    var li: usize = 1;
    while (li < tree.layers.items.len) : (li += 1) {
        const lower = &tree.layers.items[li - 1];
        const upper = &tree.layers.items[li];
        const lower_used = BW.bitIdInWord(lower.bits_count);
        const lower_last: usize = lower.activity.items.len - 1;
        var j: u32 = 0;
        while (j < upper.bits_count) : (j += 1) {
            const w: Word = lower.activity.items[j];
            const m: Word = lower.mixed.items[j];
            const valid: Word = if (j == lower_last and lower_used != 0) BW.maskStart(lower_used) else BW.max_value;
            var want: L.State = undefined;
            if (li == 1) {
                const lw = w & valid;
                want = if (lw == 0) .inactive else if (lw == valid) .active else .deep_mixed;
            } else {
                var first: L.State = .inactive;
                var seen = false;
                var uniform = true;
                var b: u32 = 0;
                while (b < BW.word_type_bits) : (b += 1) {
                    const bit: Word = @as(Word, 1) << @truncate(b);
                    if (valid & bit == 0) continue;
                    const cur = L.State.fromBits(
                        if ((w & bit) != 0) BitState.active else BitState.inactive,
                        if ((m & bit) != 0) BitState.active else BitState.inactive,
                    );
                    if (!seen) {
                        first = cur;
                        seen = true;
                    } else if (cur != first) {
                        uniform = false;
                    }
                }
                // У хранимого слова всегда есть >= 1 валидный бит.
                want = if (uniform) first else .mixed;
            }
            const uwid = BW.bitToWordId(j);
            const ubit = BW.bitIdInWord(j);
            const got = L.State.fromBits(
                BW.readBitState(upper.activity.items[uwid], ubit),
                BW.readBitState(upper.mixed.items[uwid], ubit),
            );
            try t.expectEqual(want, got);
        }
    }
}

fn treeLayerState(comptime Word: type, tree: *const BitTree(Word), li: usize, j: u32) Layer(Word).State {
    const BW = bit_word.BitWord(Word);
    const layer = &tree.layers.items[li];
    return Layer(Word).State.fromBits(
        BW.readBitState(layer.activity.items[BW.bitToWordId(j)], BW.bitIdInWord(j)),
        BW.readBitState(layer.mixed.items[BW.bitToWordId(j)], BW.bitIdInWord(j)),
    );
}

test "BitTree summary rule: unanimity + layer1 deep" {
    // Слово 0: все inactive; слово 1: все active.
    {
        var tree = BitTree(u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 128, .inactive);
        tree.setWord(1, std.math.maxInt(u64), std.math.maxInt(u64));
        try t.expectEqual(Layer(u64).State.inactive, treeLayerState(u64, &tree, 1, 0));
        try t.expectEqual(Layer(u64).State.active, treeLayerState(u64, &tree, 1, 1));
    }
    // [half/half] на слое 1 -> deep_mixed (никогда mixed).
    {
        var tree = BitTree(u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 128, .inactive);
        tree.setWord(0, 0xFFFF_FFFF, 0xFFFF_FFFF);
        try t.expectEqual(Layer(u64).State.deep_mixed, treeLayerState(u64, &tree, 1, 0));
        try t.expectEqual(Layer(u64).State.inactive, treeLayerState(u64, &tree, 1, 1));
    }
    // Слой 2 над [active, inactive] детьми -> mixed (не deep).
    // Слой 2 над all-deep детьми -> deep_mixed.
    {
        var tree = BitTree(u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 8192, .inactive);
        // Нижние слова 0..31 active, остальные inactive: биты слоя 1
        // 0..31 active, 40.. inactive. Слово 0 слоя 1 смешанное
        // (active+inactive) -> бит 0 слоя 2 mixed.
        var wid: u32 = 0;
        while (wid < 32) : (wid += 1) {
            tree.setWord(wid, std.math.maxInt(u64), std.math.maxInt(u64));
        }
        try t.expectEqual(Layer(u64).State.active, treeLayerState(u64, &tree, 1, 0));
        try t.expectEqual(Layer(u64).State.inactive, treeLayerState(u64, &tree, 1, 40));
        try t.expectEqual(Layer(u64).State.mixed, treeLayerState(u64, &tree, 2, 0));
        try t.expectEqual(Layer(u64).State.inactive, treeLayerState(u64, &tree, 2, 1));
        // Каждое слово 0..63 неоднородно (ровно 1 бит) -> биты 0..63 слоя 1
        // deep, слово 0 слоя 1 единогласно deep -> бит 0 слоя 2 deep_mixed.
        var tree2 = BitTree(u64){};
        defer tree2.deinit(t.allocator);
        try tree2.resize(t.allocator, 8192, .inactive);
        var w2: u32 = 0;
        while (w2 < 64) : (w2 += 1) {
            tree2.setBit(w2 * 64, .active);
        }
        try t.expectEqual(Layer(u64).State.deep_mixed, treeLayerState(u64, &tree2, 1, 0));
        try t.expectEqual(Layer(u64).State.inactive, treeLayerState(u64, &tree2, 1, 64));
        try t.expectEqual(Layer(u64).State.deep_mixed, treeLayerState(u64, &tree2, 2, 0));
        try t.expectEqual(Layer(u64).State.inactive, treeLayerState(u64, &tree2, 2, 1));
    }
}

/// Когерентность: плоскость activity слоя 0 == слова битсета, mixed листа
/// нулевой, счётчики == независимому скану, summary валидны.
fn expectTreeCoherent(comptime Word: type, tree: *const BitTree(Word)) !void {
    const BW = bit_word.BitWord(Word);
    const L = Layer(Word);
    const n = tree.bitset.bits_count;
    const words = tree.bitset.words.items;
    if (n == 0) {
        try t.expectEqual(@as(usize, 0), words.len);
        try t.expectEqual(@as(usize, 0), tree.layers.items.len);
        return;
    }
    try t.expectEqual(words.len, tree.layers.items[0].activity.items.len);
    // Слой 0 зеркалит битсет (в валидных битах), mixed нулевой.
    var wid: usize = 0;
    while (wid < words.len) : (wid += 1) {
        const used = BW.bitIdInWord(n);
        const valid: Word = if (wid == words.len - 1 and used != 0) BW.maskStart(used) else BW.max_value;
        try t.expectEqual(words[wid] & valid, tree.layers.items[0].activity.items[wid] & valid);
        try t.expectEqual(@as(Word, 0), tree.layers.items[0].mixed.items[wid] & valid);
    }
    // Счётчики каждого слоя == независимому скану.
    var k: usize = 0;
    while (k < tree.layers.items.len) : (k += 1) {
        const layer = &tree.layers.items[k];
        var scanned = [4]u32{ 0, 0, 0, 0 };
        var i: u32 = 0;
        while (i < layer.bits_count) : (i += 1) {
            const w2: usize = @intCast(BW.bitToWordId(i));
            const in_w = BW.bitIdInWord(i);
            const a = BW.readBitState(layer.activity.items[w2], in_w);
            const m = BW.readBitState(layer.mixed.items[w2], in_w);
            scanned[@intFromEnum(L.State.fromBits(a, m))] += 1;
        }
        try t.expectEqualSlices(u32, &scanned, &layer.state_counters);
    }
    // Лист: active == счётчик битсета, mixed/deep пустые.
    try t.expectEqual(tree.bitset.active_bits_counter, tree.layers.items[0].state_counters[L.State.active_u32]);
    try t.expectEqual(@as(u32, 0), tree.layers.items[0].state_counters[L.State.mixed_u32]);
    try t.expectEqual(@as(u32, 0), tree.layers.items[0].state_counters[L.State.deep_mixed_u32]);
    try expectSummariesValid(Word, tree);
}

fn treeStepCheckOne(comptime Word: type, n: u32, active_every: u32, active_offset: u32) !void {
    const BW = bit_word.BitWord(Word);
    var tree = BitTree(Word){};
    defer tree.deinit(t.allocator);
    try tree.resize(t.allocator, n, .inactive);
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        if (active_every != 0 and (i + active_offset) % active_every == 0) {
            tree.setBit(i, .active);
        }
    }
    try expectTreeCoherent(Word, &tree);

    // Порча паддинга битсета напрямую: step идёт через битсет и маску.
    if (n > 0) {
        const used = BW.bitIdInWord(n);
        const valid: Word = if (used == 0) std.math.maxInt(Word) else BW.maskStart(used);
        tree.bitset.words.items[tree.bitset.words.items.len - 1] |= ~valid;
    }

    var ctx = TreeStepIds{};
    try t.expect(treeStepCollectAll(Word, &tree, &ctx));
    var exp_a: [512]u32 = undefined;
    var exp_i: [512]u32 = undefined;
    const n_a = treeOracleIds(Word, &tree, .active, &exp_a);
    const n_i = treeOracleIds(Word, &tree, .inactive, &exp_i);
    try t.expectEqualSlices(u32, exp_a[0..n_a], ctx.active[0..ctx.na]);
    try t.expectEqualSlices(u32, exp_i[0..n_i], ctx.inactive[0..ctx.ni]);
    try t.expectEqual(n, @as(u32, @intCast(ctx.na + ctx.ni)));
}

test "BitTree step: counts vs oracle + corners" {
    const sizes = [_]u32{ 0, 1, 2, 63, 64, 65, 70, 127, 128, 129, 200, 300 };
    const mods = [_]u32{ 0, 1, 2, 3, 7, 64 };
    for (sizes) |n| {
        for (mods) |m| {
            try treeStepCheckOne(u64, n, m, 0);
            try treeStepCheckOne(u8, n, m, 1);
        }
    }
}

test "BitTree step: early exit" {
    // Остановка в bulk-проходе первого однородного региона.
    {
        var tree = BitTree(u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 70, .active);
        const It = BitTree(u64).Iterator(*TreeStepIds, treePushA, treePushI);
        var ctx = TreeStepIds{ .stop_after = 3 };
        try t.expect(!It.step(.{ .tree = &tree, .context = &ctx }));
        try t.expectEqual(@as(usize, 3), ctx.na);
        try t.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, ctx.active[0..ctx.na]);
    }
    // Остановка во втором регионе (смешанное дерево).
    {
        var tree = BitTree(u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 130, .inactive);
        tree.setBit(100, .active);
        const It = BitTree(u64).Iterator(*TreeStepIds, treePushA, treePushI);
        var ctx = TreeStepIds{ .stop_after = 65 };
        // 64 inactive бита слова 0, затем бит 100.
        try t.expect(!It.step(.{ .tree = &tree, .context = &ctx }));
        try t.expectEqual(@as(usize, 64), ctx.ni);
        try t.expectEqual(@as(usize, 1), ctx.na);
        try t.expectEqual(@as(u32, 100), ctx.active[0]);
    }
    // stop_after = 1: ровно один вызов и остановка.
    {
        var tree = BitTree(u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 10, .inactive);
        const It2 = BitTree(u64).Iterator(*TreeStepIds, treePushA, treePushI);
        var ctx = TreeStepIds{ .stop_after = 1 };
        try t.expect(!It2.step(.{ .tree = &tree, .context = &ctx }));
        try t.expectEqual(@as(usize, 1), ctx.ni + ctx.na);
    }
}

test "BitTree step: null side skips opposite uniform regions" {
    // Только active: однородно-inactive регионы не дают вызовов,
    // бит 100 находится через спуск.
    {
        var tree = BitTree(u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 130, .inactive);
        tree.setBit(100, .active);
        const It = BitTree(u64).Iterator(*TreeStepIds, treePushA, null);
        var ctx = TreeStepIds{};
        try t.expect(It.step(.{ .tree = &tree, .context = &ctx }));
        try t.expectEqualSlices(u32, &[_]u32{100}, ctx.active[0..ctx.na]);
        try t.expectEqual(@as(usize, 0), ctx.ni);
    }
    // Только inactive: бит 100 (active) не виден, остальные 129 на месте.
    {
        var tree = BitTree(u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 130, .inactive);
        tree.setBit(100, .active);
        const It = BitTree(u64).Iterator(*TreeStepIds, null, treePushI);
        var ctx = TreeStepIds{};
        try t.expect(It.step(.{ .tree = &tree, .context = &ctx }));
        try t.expectEqual(@as(usize, 129), ctx.ni);
        try t.expectEqual(@as(usize, 0), ctx.na);
        try t.expectEqual(@as(u32, 0), ctx.inactive[0]);
        try t.expectEqual(@as(u32, 129), ctx.inactive[128]);
    }
}

test "BitTree pyramid shape + coherence under ops" {
    // Форма пирамиды: [70] -> [70, 2, 1]; [64] -> [64, 1]; [1] -> [1]; [0] -> [].
    {
        var tree = BitTree(u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 70, .inactive);
        try t.expectEqual(@as(usize, 3), tree.layers.items.len);
        try t.expectEqual(@as(u32, 70), tree.layers.items[0].bits_count);
        try t.expectEqual(@as(u32, 2), tree.layers.items[1].bits_count);
        try t.expectEqual(@as(u32, 1), tree.layers.items[2].bits_count);
        try expectTreeCoherent(u64, &tree);
    }
    {
        var tree = BitTree(u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 64, .active);
        try t.expectEqual(@as(usize, 2), tree.layers.items.len);
        try expectTreeCoherent(u64, &tree);
        try tree.resize(t.allocator, 10, .inactive);
        try t.expectEqual(@as(usize, 2), tree.layers.items.len);
        try t.expectEqual(@as(u32, 10), tree.layers.items[0].bits_count);
        try t.expectEqual(@as(u32, 1), tree.layers.items[1].bits_count);
        try expectTreeCoherent(u64, &tree);
        try tree.resize(t.allocator, 0, .inactive);
        try t.expectEqual(@as(usize, 0), tree.layers.items.len);
        try t.expectEqual(@as(usize, 0), tree.bitset.words.items.len);
    }
    // Смесь операций: флипы на границах слов + setWord с частичной маской.
    {
        var tree = BitTree(u8){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 20, .inactive);
        tree.setBit(0, .active);
        tree.setBit(7, .active);
        tree.setBit(8, .active);
        tree.setBit(19, .active);
        tree.setWord(1, 0xF0, 0x0F);
        tree.setBit(8, .inactive);
        try expectTreeCoherent(u8, &tree);
        var ctx = TreeStepIds{};
        try t.expect(treeStepCollectAll(u8, &tree, &ctx));
        try t.expectEqual(@as(u32, 20), tree.bitset.bits_count);
        try t.expectEqual(tree.bitset.active_bits_counter, @as(u32, @intCast(ctx.na)));
    }
}

// ================= Глубокие пирамиды, прочие ширины, многоуровневый shrink =================
// Totals-харнес без хранения id (для размеров больше буфера [512]).

const StepTotals = struct {
    na: usize = 0,
    ni: usize = 0,
};

fn totalPushA(ctx: *StepTotals, bit_id: u32) bool {
    _ = bit_id;
    ctx.na += 1;
    return true;
}

fn totalPushI(ctx: *StepTotals, bit_id: u32) bool {
    _ = bit_id;
    ctx.ni += 1;
    return true;
}

fn treeStepTotals(comptime Word: type, tree: *BitTree(Word), ctx: *StepTotals) bool {
    const It = BitTree(Word).Iterator(*StepTotals, totalPushA, totalPushI);
    return It.step(.{ .tree = tree, .context = ctx });
}

/// Эталонные итоги побитовым сканом битсета (без хранения).
fn treeOracleTotals(comptime Word: type, tree: *const BitTree(Word)) [2]u64 {
    const BW = bit_word.BitWord(Word);
    var out = [2]u64{ 0, 0 };
    var i: u32 = 0;
    while (i < tree.bitset.bits_count) : (i += 1) {
        const wid: usize = @intCast(BW.bitToWordId(i));
        if (BW.readBitState(tree.bitset.words.items[wid], BW.bitIdInWord(i)) == .active) {
            out[0] += 1;
        } else {
            out[1] += 1;
        }
    }
    return out;
}

fn treeExpectTotals(comptime Word: type, tree: *BitTree(Word)) !void {
    var ctx = StepTotals{};
    try t.expect(treeStepTotals(Word, tree, &ctx));
    const want = treeOracleTotals(Word, tree);
    try t.expectEqual(want[0], ctx.na);
    try t.expectEqual(want[1], ctx.ni);
    try t.expectEqual(@as(u64, tree.bitset.bits_count), ctx.na + ctx.ni);
}

test "BitTree deep pyramid 4+ levels" {
    // u64 20000 бит: слои [20000, 313, 5, 1], разреженный паттерн.
    {
        var tree = BitTree(u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 20000, .inactive);
        try t.expectEqual(@as(usize, 4), tree.layers.items.len);
        try t.expectEqual(@as(u32, 20000), tree.layers.items[0].bits_count);
        try t.expectEqual(@as(u32, 313), tree.layers.items[1].bits_count);
        try t.expectEqual(@as(u32, 5), tree.layers.items[2].bits_count);
        try t.expectEqual(@as(u32, 1), tree.layers.items[3].bits_count);
        var i: u32 = 0;
        while (i < 20000) : (i += 997) {
            tree.setBit(i, .active);
        }
        try expectTreeCoherent(u64, &tree);
        try treeExpectTotals(u64, &tree);
    }
    // u64 20000 плотный: все active, каждый 3-й сброшен.
    {
        var tree = BitTree(u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 20000, .active);
        var i: u32 = 0;
        while (i < 20000) : (i += 3) {
            tree.setBit(i, .inactive);
        }
        try expectTreeCoherent(u64, &tree);
        try treeExpectTotals(u64, &tree);
        try t.expectEqual(@as(u32, 20000 - 6667), tree.bitset.active_bits_counter);
    }
    // u64 100000 бит: слои [100000, 1563, 25, 1], глубина 4+.
    {
        var tree = BitTree(u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 100000, .inactive);
        try t.expectEqual(@as(usize, 4), tree.layers.items.len);
        try t.expectEqual(@as(u32, 100000), tree.layers.items[0].bits_count);
        try t.expectEqual(@as(u32, 1563), tree.layers.items[1].bits_count);
        try t.expectEqual(@as(u32, 25), tree.layers.items[2].bits_count);
        try t.expectEqual(@as(u32, 1), tree.layers.items[3].bits_count);
        var k: usize = 0;
        while (k < 2000) : (k += 1) {
            tree.setBit(@truncate((7 + k * 100003) % 100000), .active);
        }
        try expectTreeCoherent(u64, &tree);
        try treeExpectTotals(u64, &tree);
    }
}

test "BitTree u16/u32 spot checks" {
    const sizes = [_]u32{ 0, 1, 2, 15, 16, 17, 31, 32, 33, 64, 100, 129, 300 };
    const mods = [_]u32{ 0, 1, 2, 5, 16 };
    for (sizes) |n| {
        for (mods) |m| {
            try treeStepCheckOne(u16, n, m, 0);
            try treeStepCheckOne(u16, n, m, 3);
            try treeStepCheckOne(u32, n, m, 0);
            try treeStepCheckOne(u32, n, m, 1);
        }
    }
    // Когерентность u16/u32 после смеси операций.
    {
        var tree = BitTree(u16){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 100, .inactive);
        tree.setBit(0, .active);
        tree.setBit(15, .active);
        tree.setBit(16, .active);
        tree.setBit(99, .active);
        tree.setWord(3, 0xFF00, 0x00FF);
        try expectTreeCoherent(u16, &tree);
        try treeExpectTotals(u16, &tree);
    }
    {
        var tree = BitTree(u32){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 100, .active);
        tree.setBit(31, .inactive);
        tree.setBit(32, .inactive);
        tree.setWord(2, 0, 0xFFFF_FFFF);
        try expectTreeCoherent(u32, &tree);
        try treeExpectTotals(u32, &tree);
    }
}

test "BitTree multi-level shrink/grow" {
    var tree = BitTree(u64){};
    defer tree.deinit(t.allocator);
    // 20000 active: глубина 4.
    try tree.resize(t.allocator, 20000, .active);
    try t.expectEqual(@as(usize, 4), tree.layers.items.len);
    try t.expectEqual(@as(u32, 20000), tree.bitset.active_bits_counter);
    try expectTreeCoherent(u64, &tree);
    // Shrink 20000 -> 10 одним вызовом: глубина 4 -> 2, первые 10 active.
    try tree.resize(t.allocator, 10, .inactive);
    try t.expectEqual(@as(usize, 2), tree.layers.items.len);
    try t.expectEqual(@as(u32, 10), tree.layers.items[0].bits_count);
    try t.expectEqual(@as(u32, 1), tree.layers.items[1].bits_count);
    try t.expectEqual(@as(u32, 10), tree.bitset.active_bits_counter);
    try expectTreeCoherent(u64, &tree);
    try treeExpectTotals(u64, &tree);
    // Grow 10 -> 5000 одним вызовом: глубина 2 -> 4.
    try tree.resize(t.allocator, 5000, .active);
    try t.expectEqual(@as(usize, 4), tree.layers.items.len);
    try t.expectEqual(@as(u32, 5000), tree.layers.items[0].bits_count);
    try t.expectEqual(@as(u32, 79), tree.layers.items[1].bits_count);
    try t.expectEqual(@as(u32, 2), tree.layers.items[2].bits_count);
    try t.expectEqual(@as(u32, 1), tree.layers.items[3].bits_count);
    try t.expectEqual(@as(u32, 5000), tree.bitset.active_bits_counter);
    try expectTreeCoherent(u64, &tree);
    try treeExpectTotals(u64, &tree);
    // Shrink в 0 и regrow: слои исчезают и создаются заново.
    try tree.resize(t.allocator, 0, .inactive);
    try t.expectEqual(@as(usize, 0), tree.layers.items.len);
    try t.expectEqual(@as(usize, 0), tree.bitset.words.items.len);
    try tree.resize(t.allocator, 64, .inactive);
    try t.expectEqual(@as(usize, 2), tree.layers.items.len);
    try expectTreeCoherent(u64, &tree);
    try treeExpectTotals(u64, &tree);
}
