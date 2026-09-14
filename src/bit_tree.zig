/// BitTree with dual-mask hierarchy for layer-by-layer slice generation.
/// Level 0 holds raw bits (64 per u64). Levels >=1 hold pairs:
/// state word (1 == uniform active) + mixed word (1 == mixed region).
/// One word pair covers 64 child regions, fanout 64.
const std = @import("std");
/// Memory allocator type used for all tree allocations.
const Allocator = std.mem.Allocator;
/// Shared utilities module import.
const utilities = @import("utilities.zig");

/// 64-byte aligned unmanaged array list constructor.
const ListA64 = utilities.ListA64;

/// State of a single bit in the bitset.
pub const BitState = enum(u1) {
    /// Bit is cleared.
    inactive = 0,
    /// Bit is set.
    active = 1,
};

/// One hierarchy level: two parallel arrays.
/// L0 uses only `state` (leaf bits), `mixed` stays empty.
/// L>=1 uses both, same length, word i of each forms a pair.
const Level = struct {
    state: ListA64(u64) = .empty,
    mixed: ListA64(u64) = .empty,

    fn deinit(self: *Level, allocator: Allocator) void {
        self.state.deinit(allocator);
        self.mixed.deinit(allocator);
        self.* = undefined;
    }
};

/// Bits per leaf word (level 0).
pub const leaf_fanout: u32 = 64;
/// Regions per summary word (levels >=1): 64 slots per u64 mask.
pub const node_fanout: u32 = 64;

/// Maximum level count (headroom for u32 indices, need 7).
const max_levels: usize = 8;

/// Word with every bit set.
const all_ones: u64 = std.math.maxInt(u64);

/// Counts leaf words needed to hold the given bit count.
inline fn leafWordsFor(bits: u32) usize {
    return std.math.divCeil(u32, bits, leaf_fanout) catch unreachable;
}

/// Counts parent summary words needed to cover the given child word count.
inline fn parentWordsFor(child_words: usize) usize {
    return std.math.divCeil(usize, child_words, node_fanout) catch unreachable;
}

/// Counts hierarchy levels needed for the given bit count.
fn depthForBits(bits: u32) usize {
    if (bits == 0) return 0;
    var depth: usize = 1;
    var words: usize = leafWordsFor(bits);
    while (words > 1) {
        words = parentWordsFor(words);
        depth += 1;
    }
    return depth;
}

/// Counts state/mixed words on one level for the given bit count.
/// L0 returns leaf words, L>=1 returns summary words.
fn wordsForLevel(level: usize, bits: u32) usize {
    if (bits == 0) return 0;
    var words: usize = leafWordsFor(bits);
    var l: usize = 0;
    while (l < level) : (l += 1) {
        words = parentWordsFor(words);
    }
    return words;
}

/// Bits covered by one id at the given layer: 64^layer.
inline fn spanBitsForLayer(layer: u32) u64 {
    // layer < max_levels (<=7), shift <= 42, fits u64.
    const s: u6 = @intCast(6 * layer);
    return @as(u64, 1) << s;
}

/// Total ids at the given layer: ceil(total_bits / span).
inline fn totalIdsForLayer(total_bits: u32, layer: u32) u64 {
    if (total_bits == 0) return 0;
    const span: u64 = spanBitsForLayer(layer);
    return (@as(u64, total_bits) + span - 1) / span;
}

/// Valid-slot mask for 0..64 valid slots.
inline fn slotsValidMask(valid: u64) u64 {
    if (valid == 0) return 0;
    if (valid >= 64) return all_ones;
    const v: u6 = @intCast(valid);
    return (@as(u64, 1) << v) - 1;
}

/// Builds the valid-bit mask for the last leaf word.
inline fn lastLeafMask(total_bits: u32) u64 {
    const r: u6 = @intCast(total_bits & 63);
    if (r == 0) return all_ones;
    return (@as(u64, 1) << r) - 1;
}

/// Builds the mask of word bits covered by a half-open bit range.
fn rangeMask(word_idx: usize, range_start: u32, range_end: u64) u64 {
    const ws: u64 = @as(u64, word_idx) * leaf_fanout;
    var lo: u32 = 0;
    if (range_start > ws) lo = @intCast(@as(u64, range_start) - ws);
    var hi: u32 = leaf_fanout;
    if (range_end - ws < leaf_fanout) hi = @intCast(range_end - ws);
    const width: u32 = hi - lo;
    if (width == leaf_fanout) return all_ones;
    const lo_s: u6 = @intCast(lo);
    const w_s: u6 = @intCast(width);
    return ((@as(u64, 1) << w_s) - 1) << lo_s;
}

/// Per-polarity output granularity for LayerBitsIterator.
/// Each field is independent: uniform callbacks may emit raw bit ids (0)
/// while mixed descends at coarser granularity, or all three may be equal.
pub const OutLayers = struct {
    active: u32,
    inactive: u32,
    mixed: u32,
};

/// Bitset with a dual-mask hierarchy, scanned one layer at a time.
pub const BitTree = struct {
    /// Short alias of the enclosing type.
    const Self = @This();

    /// Number of addressable bits in the bitset.
    total_bits: u32 = 0,
    /// Number of set bits. Updated by every mutating operation.
    active_count: u32 = 0,
    /// Hierarchy levels. Index 0 is the leaf bitset.
    levels: ListA64(Level) = .empty,

    /// Empty tree with no bits and no allocated levels.
    pub const empty: Self = .{};

    /// Releases every level buffer and invalidates the tree.
    pub fn deinit(self: *Self, allocator: Allocator) void {
        for (self.levels.items) |*lvl| {
            lvl.deinit(allocator);
        }
        self.levels.deinit(allocator);
        self.* = undefined;
    }

    /// Reports the number of addressable bits.
    pub inline fn totalBitsCount(self: *const Self) u32 {
        return self.total_bits;
    }

    /// Counts bits in the given state in O(1).
    pub inline fn count(self: *const Self, state: BitState) u32 {
        return switch (state) {
            .active => self.active_count,
            .inactive => self.total_bits - self.active_count,
        };
    }

    /// Reads a single bit as bool. True == active.
    pub fn getBit(self: *const Self, id: u32) bool {
        std.debug.assert(id < self.total_bits);
        const w: u64 = self.levels.items[0].state.items[id >> 6];
        const s: u6 = @intCast(id & 63);
        return ((w >> s) & 1) == 1;
    }

    /// Reads a single bit as BitState.
    pub fn get(self: *const Self, bit: u32) BitState {
        return if (self.getBit(bit)) .active else .inactive;
    }

    /// Writes a single bit from bool. Refreshes the ancestor chain.
    pub fn setBit(self: *Self, id: u32, value: bool) void {
        std.debug.assert(id < self.total_bits);
        std.debug.assert(self.levels.items.len > 0);
        const word_idx: usize = id >> 6;
        const shift: u6 = @intCast(id & 63);
        const mask: u64 = @as(u64, 1) << shift;
        const leaf: *u64 = &self.levels.items[0].state.items[word_idx];
        const was_active = (leaf.* & mask) != 0;
        if (was_active == value) return;
        if (value) {
            leaf.* |= mask;
            self.active_count += 1;
        } else {
            leaf.* &= ~mask;
            self.active_count -= 1;
        }
        self.propagateFromLeafWord(word_idx);
    }

    /// Sets a single bit from BitState.
    pub fn set(self: *Self, bit: u32, state: BitState) void {
        self.setBit(bit, state == .active);
    }

    /// Summarizes one leaf word into (state_bit, mixed_bit) for its parent slot.
    const SlotSummary = struct { s: bool, m: bool };
    inline fn summarizeLeafBits(word: u64, valid_mask: u64) SlotSummary {
        const w = word & valid_mask;
        if (w == 0) return .{ .s = false, .m = false };
        if (w == valid_mask) return .{ .s = true, .m = false };
        return .{ .s = false, .m = true };
    }

    /// Summarizes one child summary word (pair) into (state_bit, mixed_bit).
    /// Child covers up to 64 slots, only `valid` of them are real.
    inline fn summarizeChildWord(child_state: u64, child_mixed: u64, valid: u32) SlotSummary {
        const vm = slotsValidMask(valid);
        const m = child_mixed & vm;
        if (m != 0) return .{ .s = false, .m = true };
        const s = child_state & vm;
        if (s == 0) return .{ .s = false, .m = false };
        if (s == vm) return .{ .s = true, .m = false };
        return .{ .s = false, .m = true };
    }

    /// Writes one slot bit pair into parent words. Returns true when changed.
    inline fn writeSlot(parent_state: *u64, parent_mixed: *u64, slot: u32, s: bool, m: bool) bool {
        const bit: u64 = @as(u64, 1) << @as(u6, @intCast(slot));
        const old_s = (parent_state.* & bit) != 0;
        const old_m = (parent_mixed.* & bit) != 0;
        if (old_s == s and old_m == m) return false;
        if (s) parent_state.* |= bit else parent_state.* &= ~bit;
        if (m) parent_mixed.* |= bit else parent_mixed.* &= ~bit;
        return true;
    }

    /// Refreshes the ancestor chain of one leaf word after its content changed.
    fn propagateFromLeafWord(self: *Self, leaf_word_idx: usize) void {
        if (self.levels.items.len <= 1) return;
        var child_idx: usize = leaf_word_idx;
        // Summary of leaf word for level 1 slot.
        var cur: SlotSummary = blk: {
            const leaves = self.levels.items[0].state.items;
            const mask: u64 = if (child_idx + 1 == leaves.len) lastLeafMask(self.total_bits) else all_ones;
            break :blk summarizeLeafBits(leaves[child_idx], mask);
        };
        var lvl: usize = 1;
        while (lvl < self.levels.items.len) : (lvl += 1) {
            const p_idx: usize = child_idx / node_fanout;
            const slot: u32 = @intCast(child_idx % node_fanout);
            const lvl_items_state = self.levels.items[lvl].state.items;
            const lvl_items_mixed = self.levels.items[lvl].mixed.items;
            const changed = writeSlot(&lvl_items_state[p_idx], &lvl_items_mixed[p_idx], slot, cur.s, cur.m);
            if (!changed) return;
            // Summarize the just-written parent word for the next level.
            const total_child_ids: u64 = totalIdsForLayer(self.total_bits, @intCast(lvl));
            // Child word index for next step is p_idx; but to summarize word p_idx
            // we need its valid slot count.
            const base: u64 = @as(u64, p_idx) * node_fanout;
            const remain: u64 = if (base >= total_child_ids) 0 else total_child_ids - base;
            const valid: u32 = @intCast(@min(remain, @as(u64, node_fanout)));
            cur = summarizeChildWord(lvl_items_state[p_idx], lvl_items_mixed[p_idx], valid);
            child_idx = p_idx;
        }
    }

    /// Sets a bit range to one state.
    pub fn setRange(self: *Self, start: u32, len: u32, state: BitState) void {
        if (len == 0) return;
        std.debug.assert(self.levels.items.len > 0);
        std.debug.assert(@as(u64, start) + len <= self.total_bits);
        const end: u64 = @as(u64, start) + len;
        const first_word: usize = @intCast(start >> 6);
        const last_word: usize = @intCast((end - 1) >> 6);
        const leaves = self.levels.items[0].state.items;
        const want_one = state == .active;
        var w: usize = first_word;
        var changed: u32 = 0;
        while (w <= last_word) : (w += 1) {
            const m = rangeMask(w, start, end);
            const old = leaves[w];
            const new = if (want_one) old | m else old & ~m;
            if (new != old) {
                leaves[w] = new;
                const added: u32 = @intCast(@popCount(new & ~old));
                const removed: u32 = @intCast(@popCount(old & ~new));
                self.active_count += added;
                self.active_count -= removed;
                changed += 1;
            }
        }
        if (changed == 0) return;
        if (changed > node_fanout * 4) {
            self.rebuildSummaries();
        } else {
            w = first_word;
            while (w <= last_word) : (w += 1) {
                self.propagateFromLeafWord(w);
            }
        }
    }

    /// Fills the whole bitset with one state and rebuilds every summary.
    pub fn clear(self: *Self, state: BitState) void {
        if (self.total_bits == 0) return;
        const leaves = self.levels.items[0].state.items;
        @memset(leaves, if (state == .active) all_ones else 0);
        if (self.total_bits & 63 != 0) {
            leaves[leaves.len - 1] &= lastLeafMask(self.total_bits);
        }
        self.active_count = if (state == .active) self.total_bits else 0;
        self.rebuildSummaries();
    }

    /// Precise resize.
    pub fn resize(self: *Self, allocator: Allocator, new_bits_count: u32, new_bits_state: BitState) !void {
        try self.resizeImpl(allocator, new_bits_count, new_bits_state, .precise);
    }

    /// Resize that retains capacity.
    pub fn resizeRetainingCapacity(self: *Self, allocator: Allocator, new_bits_count: u32, new_bits_state: BitState) !void {
        try self.resizeImpl(allocator, new_bits_count, new_bits_state, .retaining);
    }

    const ResizeMode = enum { precise, retaining };

    fn resizeImpl(self: *Self, allocator: Allocator, new_bits_count: u32, new_bits_state: BitState, comptime mode: ResizeMode) !void {
        const old_bits = self.total_bits;
        if (new_bits_count == old_bits) {
            if (mode == .precise) {
                for (self.levels.items) |*lvl| {
                    if (lvl.state.capacity != lvl.state.items.len) lvl.state.shrinkAndFree(allocator, lvl.state.items.len);
                    if (lvl.mixed.capacity != lvl.mixed.items.len) lvl.mixed.shrinkAndFree(allocator, lvl.mixed.items.len);
                }
                if (self.levels.capacity != self.levels.items.len) {
                    self.levels.shrinkAndFree(allocator, self.levels.items.len);
                }
            }
            return;
        }

        const old_depth: usize = self.levels.items.len;
        const old_leaf_words: usize = if (old_depth > 0) self.levels.items[0].state.items.len else 0;

        var removed_active: u32 = 0;
        if (new_bits_count < old_bits and old_leaf_words > 0) {
            removed_active = self.countRangeActive(new_bits_count, old_bits - new_bits_count);
        }

        const need_depth: usize = depthForBits(new_bits_count);
        std.debug.assert(need_depth <= max_levels);

        if (need_depth > old_depth) {
            switch (mode) {
                .precise => try self.levels.ensureTotalCapacityPrecise(allocator, need_depth),
                .retaining => try self.levels.ensureTotalCapacity(allocator, need_depth),
            }
        }
        var sizes: [max_levels]usize = undefined;
        var l: usize = 0;
        while (l < need_depth) : (l += 1) {
            sizes[l] = wordsForLevel(l, new_bits_count);
            if (l < old_depth) {
                switch (mode) {
                    .precise => {
                        try self.levels.items[l].state.ensureTotalCapacityPrecise(allocator, sizes[l]);
                        if (l == 0) {
                            // L0 mixed stays empty.
                        } else {
                            try self.levels.items[l].mixed.ensureTotalCapacityPrecise(allocator, sizes[l]);
                        }
                    },
                    .retaining => {
                        try self.levels.items[l].state.ensureTotalCapacity(allocator, sizes[l]);
                        if (l != 0) {
                            try self.levels.items[l].mixed.ensureTotalCapacity(allocator, sizes[l]);
                        }
                    },
                }
            }
        }
        var new_levels: [max_levels]Level = undefined;
        var new_count: usize = 0;
        errdefer {
            var i: usize = 0;
            while (i < new_count) : (i += 1) {
                new_levels[i].deinit(allocator);
            }
        }
        l = old_depth;
        while (l < need_depth) : (l += 1) {
            var lvl: Level = .{};
            lvl.state = try ListA64(u64).initCapacity(allocator, sizes[l]);
            if (l != 0) {
                lvl.mixed = try ListA64(u64).initCapacity(allocator, sizes[l]);
            }
            new_levels[new_count] = lvl;
            new_count += 1;
        }

        l = 0;
        while (l < old_depth and l < need_depth) : (l += 1) {
            try self.levels.items[l].state.resize(allocator, sizes[l]);
            if (l != 0) {
                try self.levels.items[l].mixed.resize(allocator, sizes[l]);
            } else {
                // Keep L0 mixed empty.
                if (self.levels.items[l].mixed.items.len != 0) {
                    try self.levels.items[l].mixed.resize(allocator, 0);
                }
            }
        }
        l = 0;
        while (l < new_count) : (l += 1) {
            try new_levels[l].state.resize(allocator, sizes[old_depth + l]);
            @memset(new_levels[l].state.items, 0);
            if (old_depth + l != 0) {
                try new_levels[l].mixed.resize(allocator, sizes[old_depth + l]);
                @memset(new_levels[l].mixed.items, 0);
            }
            self.levels.appendAssumeCapacity(new_levels[l]);
        }
        new_count = 0;

        self.total_bits = new_bits_count;

        var added_active: u32 = 0;
        if (new_bits_count > old_bits) {
            const leaves = self.levels.items[0].state.items;
            if (old_leaf_words < leaves.len) {
                @memset(leaves[old_leaf_words..], if (new_bits_state == .active) all_ones else 0);
            }
            if (new_bits_state == .active) {
                if (old_bits > 0 and old_bits & 63 != 0) {
                    const tail_idx: usize = old_leaf_words - 1;
                    const word_end: u64 = (@as(u64, tail_idx) + 1) * leaf_fanout;
                    const fill_to: u64 = @min(@as(u64, new_bits_count), word_end);
                    leaves[tail_idx] |= rangeMask(tail_idx, old_bits, fill_to);
                }
                added_active = new_bits_count - old_bits;
            }
            if (new_bits_count & 63 != 0) {
                leaves[leaves.len - 1] &= lastLeafMask(new_bits_count);
            }
        } else {
            if (new_bits_count > 0 and new_bits_count & 63 != 0) {
                const leaves = self.levels.items[0].state.items;
                leaves[leaves.len - 1] &= lastLeafMask(new_bits_count);
            }
        }
        self.active_count = self.active_count - removed_active + added_active;

        if (need_depth < old_depth) {
            var d: usize = need_depth;
            while (d < old_depth) : (d += 1) {
                self.levels.items[d].deinit(allocator);
            }
            if (mode == .precise) {
                self.levels.shrinkAndFree(allocator, need_depth);
            } else {
                self.levels.shrinkRetainingCapacity(need_depth);
            }
        } else if (mode == .precise and self.levels.capacity != self.levels.items.len) {
            self.levels.shrinkAndFree(allocator, self.levels.items.len);
        }

        self.rebuildSummaries();

        if (mode == .precise) {
            l = 0;
            while (l < need_depth) : (l += 1) {
                const lvl = &self.levels.items[l];
                if (lvl.state.capacity != lvl.state.items.len) lvl.state.shrinkAndFree(allocator, lvl.state.items.len);
                if (l != 0 and lvl.mixed.capacity != lvl.mixed.items.len) lvl.mixed.shrinkAndFree(allocator, lvl.mixed.items.len);
            }
        }
    }

    /// Counts set bits inside a bit range.
    fn countRangeActive(self: *const Self, start: u32, len: u32) u32 {
        if (len == 0) return 0;
        const leaves = self.levels.items[0].state.items;
        const end: u64 = @as(u64, start) + len;
        var w: usize = @intCast(start >> 6);
        const last_word: usize = @intCast((end - 1) >> 6);
        var acc: u32 = 0;
        while (w <= last_word) : (w += 1) {
            acc += @intCast(@popCount(leaves[w] & rangeMask(w, start, end)));
        }
        return acc;
    }

    /// Recomputes every summary bottom-up.
    fn rebuildSummaries(self: *Self) void {
        const nlevels = self.levels.items.len;
        var lvl: usize = 1;
        while (lvl < nlevels) : (lvl += 1) {
            const parent_state = self.levels.items[lvl].state.items;
            const parent_mixed = self.levels.items[lvl].mixed.items;
            const total_child_ids: u64 = totalIdsForLayer(self.total_bits, @intCast(lvl));
            var p: usize = 0;
            while (p < parent_state.len) : (p += 1) {
                var sw: u64 = 0;
                var mw: u64 = 0;
                const first_child: usize = p * node_fanout;
                const base_id: u64 = @as(u64, first_child);
                const remain: u64 = if (base_id >= total_child_ids) 0 else total_child_ids - base_id;
                const valid: u32 = @intCast(@min(remain, @as(u64, node_fanout)));
                var s: u32 = 0;
                while (s < valid) : (s += 1) {
                    const ci: usize = first_child + s;
                    const summ: SlotSummary = if (lvl == 1) blk: {
                        const leaves = self.levels.items[0].state.items;
                        const mask: u64 = if (ci + 1 == leaves.len) lastLeafMask(self.total_bits) else all_ones;
                        break :blk summarizeLeafBits(leaves[ci], mask);
                    } else blk: {
                        const cw_s: u64 = self.levels.items[lvl - 1].state.items[ci];
                        const cw_m: u64 = self.levels.items[lvl - 1].mixed.items[ci];
                        const below_total: u64 = totalIdsForLayer(self.total_bits, @intCast(lvl - 1));
                        const cbase: u64 = @as(u64, ci) * node_fanout;
                        const crem: u64 = if (cbase >= below_total) 0 else below_total - cbase;
                        const cvalid: u32 = @intCast(@min(crem, @as(u64, node_fanout)));
                        break :blk summarizeChildWord(cw_s, cw_m, cvalid);
                    };
                    if (summ.s) sw |= @as(u64, 1) << @as(u6, @intCast(s));
                    if (summ.m) mw |= @as(u64, 1) << @as(u6, @intCast(s));
                }
                parent_state[p] = sw;
                parent_mixed[p] = mw;
            }
        }
    }

    /// Full per-bit descent from root to leaves through a comptime stage chain.
    /// Output is ascending bit order: every stage scans slots ascending and mixed
    /// regions are descended synchronously (depth-first), so callbacks fire in order.
    /// Returns false when a callback stopped the walk early, true otherwise.
    /// No allocation, no output arrays: results stream straight into callbacks.
    pub fn iterateTargetBits(
        self: *const Self,
        comptime Ctx: type,
        ctx: Ctx,
        comptime on_active: ?fn (Ctx, u32) bool,
        comptime on_inactive: ?fn (Ctx, u32) bool,
    ) bool {
        if (on_active == null and on_inactive == null) return true;
        const depth: usize = self.levels.items.len;
        switch (depth) {
            0 => return true,
            1 => {
                const Leaf = BitsetIterator(Ctx, 0, on_active, on_inactive);
                return Leaf.runAll(ctx, self);
            },
            2 => {
                const Leaf = BitsetIterator(Ctx, 1, on_active, on_inactive);
                const L1 = LayerBitsIterator(Ctx, 1, 1, .{ .active = 0, .inactive = 0, .mixed = 1 }, on_active, on_inactive, Leaf.step);
                return L1.runAll(ctx, self);
            },
            3 => {
                const Leaf = BitsetIterator(Ctx, 1, on_active, on_inactive);
                const L1 = LayerBitsIterator(Ctx, 1, 2, .{ .active = 0, .inactive = 0, .mixed = 1 }, on_active, on_inactive, Leaf.step);
                const L2 = LayerBitsIterator(Ctx, 2, 2, .{ .active = 0, .inactive = 0, .mixed = 2 }, on_active, on_inactive, L1.step);
                return L2.runAll(ctx, self);
            },
            4 => {
                const Leaf = BitsetIterator(Ctx, 1, on_active, on_inactive);
                const L1 = LayerBitsIterator(Ctx, 1, 2, .{ .active = 0, .inactive = 0, .mixed = 1 }, on_active, on_inactive, Leaf.step);
                const L2 = LayerBitsIterator(Ctx, 2, 3, .{ .active = 0, .inactive = 0, .mixed = 2 }, on_active, on_inactive, L1.step);
                const L3 = LayerBitsIterator(Ctx, 3, 3, .{ .active = 0, .inactive = 0, .mixed = 3 }, on_active, on_inactive, L2.step);
                return L3.runAll(ctx, self);
            },
            5 => {
                const Leaf = BitsetIterator(Ctx, 1, on_active, on_inactive);
                const L1 = LayerBitsIterator(Ctx, 1, 2, .{ .active = 0, .inactive = 0, .mixed = 1 }, on_active, on_inactive, Leaf.step);
                const L2 = LayerBitsIterator(Ctx, 2, 3, .{ .active = 0, .inactive = 0, .mixed = 2 }, on_active, on_inactive, L1.step);
                const L3 = LayerBitsIterator(Ctx, 3, 4, .{ .active = 0, .inactive = 0, .mixed = 3 }, on_active, on_inactive, L2.step);
                const L4 = LayerBitsIterator(Ctx, 4, 4, .{ .active = 0, .inactive = 0, .mixed = 4 }, on_active, on_inactive, L3.step);
                return L4.runAll(ctx, self);
            },
            6 => {
                const Leaf = BitsetIterator(Ctx, 1, on_active, on_inactive);
                const L1 = LayerBitsIterator(Ctx, 1, 2, .{ .active = 0, .inactive = 0, .mixed = 1 }, on_active, on_inactive, Leaf.step);
                const L2 = LayerBitsIterator(Ctx, 2, 3, .{ .active = 0, .inactive = 0, .mixed = 2 }, on_active, on_inactive, L1.step);
                const L3 = LayerBitsIterator(Ctx, 3, 4, .{ .active = 0, .inactive = 0, .mixed = 3 }, on_active, on_inactive, L2.step);
                const L4 = LayerBitsIterator(Ctx, 4, 5, .{ .active = 0, .inactive = 0, .mixed = 4 }, on_active, on_inactive, L3.step);
                const L5 = LayerBitsIterator(Ctx, 5, 5, .{ .active = 0, .inactive = 0, .mixed = 5 }, on_active, on_inactive, L4.step);
                return L5.runAll(ctx, self);
            },
            7 => {
                const Leaf = BitsetIterator(Ctx, 1, on_active, on_inactive);
                const L1 = LayerBitsIterator(Ctx, 1, 2, .{ .active = 0, .inactive = 0, .mixed = 1 }, on_active, on_inactive, Leaf.step);
                const L2 = LayerBitsIterator(Ctx, 2, 3, .{ .active = 0, .inactive = 0, .mixed = 2 }, on_active, on_inactive, L1.step);
                const L3 = LayerBitsIterator(Ctx, 3, 4, .{ .active = 0, .inactive = 0, .mixed = 3 }, on_active, on_inactive, L2.step);
                const L4 = LayerBitsIterator(Ctx, 4, 5, .{ .active = 0, .inactive = 0, .mixed = 4 }, on_active, on_inactive, L3.step);
                const L5 = LayerBitsIterator(Ctx, 5, 6, .{ .active = 0, .inactive = 0, .mixed = 5 }, on_active, on_inactive, L4.step);
                const L6 = LayerBitsIterator(Ctx, 6, 6, .{ .active = 0, .inactive = 0, .mixed = 6 }, on_active, on_inactive, L5.step);
                return L6.runAll(ctx, self);
            },
            8 => {
                const Leaf = BitsetIterator(Ctx, 1, on_active, on_inactive);
                const L1 = LayerBitsIterator(Ctx, 1, 2, .{ .active = 0, .inactive = 0, .mixed = 1 }, on_active, on_inactive, Leaf.step);
                const L2 = LayerBitsIterator(Ctx, 2, 3, .{ .active = 0, .inactive = 0, .mixed = 2 }, on_active, on_inactive, L1.step);
                const L3 = LayerBitsIterator(Ctx, 3, 4, .{ .active = 0, .inactive = 0, .mixed = 3 }, on_active, on_inactive, L2.step);
                const L4 = LayerBitsIterator(Ctx, 4, 5, .{ .active = 0, .inactive = 0, .mixed = 4 }, on_active, on_inactive, L3.step);
                const L5 = LayerBitsIterator(Ctx, 5, 6, .{ .active = 0, .inactive = 0, .mixed = 5 }, on_active, on_inactive, L4.step);
                const L6 = LayerBitsIterator(Ctx, 6, 7, .{ .active = 0, .inactive = 0, .mixed = 6 }, on_active, on_inactive, L5.step);
                const L7 = LayerBitsIterator(Ctx, 7, 7, .{ .active = 0, .inactive = 0, .mixed = 7 }, on_active, on_inactive, L6.step);
                return L7.runAll(ctx, self);
            },
            else => unreachable,
        }
    }
};

/// Callback-based single-layer iterator factory.
/// Callbacks take (Ctx, u32) [mixed: (Ctx, *const BitTree, u32)] and return
/// bool (false stops the walk).
///
/// Scans words at `scan_layer`. `step(id)` treats `id` as one entry in
/// `in_layer` format (requires `in_layer >= scan_layer`) and processes the
/// covered `scan_layer` range; `runAll()` scans the whole layer.
/// Each non-null callback fires per matching id converted to its own
/// `out_layers` granularity (each requires `out <= scan_layer`).
/// Slots are visited in ascending order and mixed regions are descended
/// synchronously, so callbacks fire in ascending id order (depth-first).
/// Null callbacks prune both the call and the mask computation at comptime.
/// Returns false on the first callback that returns false, true otherwise.
/// Never allocates.
pub fn LayerBitsIterator(
    comptime Ctx: type,
    comptime scan_layer: u32,
    comptime in_layer: u32,
    comptime out_layers: OutLayers,
    comptime on_active: ?fn (Ctx, u32) bool,
    comptime on_inactive: ?fn (Ctx, u32) bool,
    comptime on_mixed: ?fn (Ctx, *const BitTree, u32) bool,
) type {
    if (in_layer < scan_layer) @compileError("in_layer must be >= scan_layer");
    if (on_active != null and out_layers.active > scan_layer) @compileError("out active must be <= scan_layer");
    if (on_inactive != null and out_layers.inactive > scan_layer) @compileError("out inactive must be <= scan_layer");
    if (on_mixed != null and out_layers.mixed > scan_layer) @compileError("out mixed must be <= scan_layer");
    if (on_active == null and on_inactive == null and on_mixed == null) @compileError("at least one callback required");
    return struct {
        inline fn emitOne(ctx: Ctx, id_scan: u64, comptime out_l: u32, comptime cb: ?fn (Ctx, u32) bool, total_out: u64) bool {
            const f = cb orelse return true;
            const d: u32 = scan_layer - out_l;
            if (d == 0) {
                if (id_scan >= total_out) return true;
                return f(ctx, @intCast(id_scan));
            }
            const sh: u6 = @intCast(6 * d);
            const base: u64 = id_scan << sh;
            const cnt: u64 = @as(u64, 1) << sh;
            // Fast path: fully covered range needs no per-element bounds check,
            // which keeps the loop vectorizable.
            if (base + cnt <= total_out) {
                var k: u64 = 0;
                while (k < cnt) : (k += 1) {
                    if (!f(ctx, @intCast(base + k))) return false;
                }
                return true;
            }
            if (base >= total_out) return true;
            var k: u64 = 0;
            while (k < cnt) : (k += 1) {
                const oid: u64 = base + k;
                if (oid >= total_out) break;
                if (!f(ctx, @intCast(oid))) return false;
            }
            return true;
        }

        inline fn emitMixedOne(ctx: Ctx, tree: *const BitTree, id_scan: u64, comptime out_l: u32, comptime cb: ?fn (Ctx, *const BitTree, u32) bool, total_out: u64) bool {
            const f = cb orelse return true;
            const d: u32 = scan_layer - out_l;
            if (d == 0) {
                if (id_scan >= total_out) return true;
                return f(ctx, tree, @intCast(id_scan));
            }
            const sh: u6 = @intCast(6 * d);
            const base: u64 = id_scan << sh;
            const cnt: u64 = @as(u64, 1) << sh;
            if (base + cnt <= total_out) {
                var k: u64 = 0;
                while (k < cnt) : (k += 1) {
                    if (!f(ctx, tree, @intCast(base + k))) return false;
                }
                return true;
            }
            if (base >= total_out) return true;
            var k: u64 = 0;
            while (k < cnt) : (k += 1) {
                const oid: u64 = base + k;
                if (oid >= total_out) break;
                if (!f(ctx, tree, @intCast(oid))) return false;
            }
            return true;
        }

        inline fn rangeMask64(lo: u64, hi: u64) u64 {
            const width: u32 = @intCast(hi - lo);
            if (width == 64) return all_ones;
            const w6: u6 = @intCast(width);
            const l6: u6 = @intCast(lo);
            return (((@as(u64, 1) << w6) - 1) << l6);
        }

        inline fn processWord(ctx: Ctx, tree: *const BitTree, w: usize, allowed: u64, t_a: u64, t_i: u64, t_m: u64) bool {
            if (allowed == 0) return true;
            const st: u64 = tree.levels.items[scan_layer].state.items[w];
            const mx: u64 = tree.levels.items[scan_layer].mixed.items[w];
            const m_all: u64 = mx & allowed;
            const a_all: u64 = st & ~mx & allowed;
            var rest: u64 = 0;
            if (on_mixed != null) rest |= m_all;
            if (on_active != null) rest |= a_all;
            if (on_inactive != null) rest |= allowed & ~m_all & ~a_all;
            while (rest != 0) {
                const s: u32 = @ctz(rest);
                rest &= rest - 1;
                const bitm: u64 = @as(u64, 1) << @as(u6, @intCast(s));
                const id_scan: u64 = @as(u64, w) * 64 + s;
                if ((m_all & bitm) != 0) {
                    if (!emitMixedOne(ctx, tree, id_scan, out_layers.mixed, on_mixed, t_m)) return false;
                } else if ((a_all & bitm) != 0) {
                    if (!emitOne(ctx, id_scan, out_layers.active, on_active, t_a)) return false;
                } else {
                    if (!emitOne(ctx, id_scan, out_layers.inactive, on_inactive, t_i)) return false;
                }
            }
            return true;
        }

        /// Processes one `in_layer` entry: the covered `scan_layer` word range.
        /// Out-of-range ids are skipped. Returns false on early callback stop.
        pub fn step(ctx: Ctx, tree: *const BitTree, id: u32) bool {
            const total = tree.total_bits;
            if (total == 0) return true;
            if (scan_layer >= tree.levels.items.len) return true;
            const total_in: u64 = totalIdsForLayer(total, in_layer);
            if (@as(u64, id) >= total_in) return true;
            const d_in: u32 = in_layer - scan_layer;
            const base: u64 = if (d_in == 0) id else @as(u64, id) << @as(u6, @intCast(6 * d_in));
            var cnt: u64 = if (d_in == 0) 1 else @as(u64, 1) << @as(u6, @intCast(6 * d_in));
            const total_scan: u64 = totalIdsForLayer(total, scan_layer);
            if (base >= total_scan) return true;
            if (base + cnt > total_scan) cnt = total_scan - base;
            const t_a: u64 = totalIdsForLayer(total, out_layers.active);
            const t_i: u64 = totalIdsForLayer(total, out_layers.inactive);
            const t_m: u64 = totalIdsForLayer(total, out_layers.mixed);
            const words: usize = tree.levels.items[scan_layer].state.items.len;
            const w_first: usize = @intCast(base >> 6);
            const w_last: usize = @intCast((base + cnt - 1) >> 6);
            var w: usize = w_first;
            while (w <= w_last) : (w += 1) {
                if (w >= words) break;
                const w_base: u64 = @as(u64, w) * 64;
                const w_end: u64 = w_base + 64;
                const lo: u64 = if (base > w_base) base - w_base else 0;
                const hi: u64 = if (base + cnt < w_end) base + cnt - w_base else 64;
                if (hi <= lo) continue;
                var allowed: u64 = rangeMask64(lo, hi);
                const trem: u64 = if (w_base >= total_scan) 0 else total_scan - w_base;
                if (trem == 0) continue;
                allowed &= slotsValidMask(@min(trem, @as(u64, 64)));
                if (!processWord(ctx, tree, w, allowed, t_a, t_i, t_m)) return false;
            }
            return true;
        }

        /// Processes every word at `scan_layer` in ascending order.
        /// Returns false on early callback stop.
        pub fn runAll(ctx: Ctx, tree: *const BitTree) bool {
            const total = tree.total_bits;
            if (total == 0) return true;
            if (scan_layer >= tree.levels.items.len) return true;
            const total_scan: u64 = totalIdsForLayer(total, scan_layer);
            const t_a: u64 = totalIdsForLayer(total, out_layers.active);
            const t_i: u64 = totalIdsForLayer(total, out_layers.inactive);
            const t_m: u64 = totalIdsForLayer(total, out_layers.mixed);
            const words: usize = tree.levels.items[scan_layer].state.items.len;
            var w: usize = 0;
            while (w < words) : (w += 1) {
                const base: u64 = @as(u64, w) * 64;
                if (base >= total_scan) break;
                const remain: u64 = total_scan - base;
                if (!processWord(ctx, tree, w, slotsValidMask(@min(remain, @as(u64, 64))), t_a, t_i, t_m)) return false;
            }
            return true;
        }
    };
}

/// Callback-based leaf iterator factory. Output is always raw bit ids (layer 0).
/// `step(id)` treats `id` as one `in_layer` entry and classifies the covered bits;
/// `runAll()` scans the whole bitmask. Ascending order, no allocation.
pub fn BitsetIterator(
    comptime Ctx: type,
    comptime in_layer: u32,
    comptime on_active: ?fn (Ctx, u32) bool,
    comptime on_inactive: ?fn (Ctx, u32) bool,
) type {
    if (on_active == null and on_inactive == null) @compileError("at least one callback required");
    return struct {
        inline fn processLeafWord(ctx: Ctx, leaves: []const u64, w: usize, allowed: u64) bool {
            if (allowed == 0) return true;
            const lw: u64 = leaves[w];
            var wanted: u64 = 0;
            if (on_active != null) wanted |= lw & allowed;
            if (on_inactive != null) wanted |= ~lw & allowed;
            const base: u64 = @as(u64, w) * 64;
            while (wanted != 0) {
                const s: u32 = @ctz(wanted);
                wanted &= wanted - 1;
                const is_active = ((lw >> @as(u6, @intCast(s))) & 1) == 1;
                if (is_active) {
                    if (on_active) |f| {
                        if (!f(ctx, @intCast(base + s))) return false;
                    }
                } else {
                    if (on_inactive) |f| {
                        if (!f(ctx, @intCast(base + s))) return false;
                    }
                }
            }
            return true;
        }

        /// Classifies one `in_layer` entry (a single bit when `in_layer == 0`).
        /// Out-of-range ids are skipped. Returns false on early callback stop.
        pub fn step(ctx: Ctx, tree: *const BitTree, id: u32) bool {
            const total = tree.total_bits;
            if (total == 0) return true;
            if (tree.levels.items.len == 0) return true;
            const total_in: u64 = totalIdsForLayer(total, in_layer);
            if (@as(u64, id) >= total_in) return true;
            const sh: u6 = @intCast(6 * in_layer);
            const base: u64 = if (in_layer == 0) id else @as(u64, id) << sh;
            var cnt: u64 = if (in_layer == 0) 1 else (@as(u64, 1) << sh);
            if (base >= total) return true;
            if (base + cnt > total) cnt = @as(u64, total) - base;
            const leaves = tree.levels.items[0].state.items;
            const w_first: usize = @intCast(base >> 6);
            const w_last: usize = @intCast((base + cnt - 1) >> 6);
            var w: usize = w_first;
            while (w <= w_last) : (w += 1) {
                if (w >= leaves.len) break;
                const w_base: u64 = @as(u64, w) * 64;
                const w_end: u64 = w_base + 64;
                const lo: u64 = if (base > w_base) base - w_base else 0;
                const hi: u64 = if (base + cnt < w_end) base + cnt - w_base else 64;
                if (hi <= lo) continue;
                const width: u32 = @intCast(hi - lo);
                var allowed: u64 = if (width == 64) all_ones else (((@as(u64, 1) << @as(u6, @intCast(width))) - 1) << @as(u6, @intCast(lo)));
                if (w + 1 == leaves.len) allowed &= lastLeafMask(total);
                if (!processLeafWord(ctx, leaves, w, allowed)) return false;
            }
            return true;
        }

        /// Classifies every bit in ascending order. Returns false on early stop.
        pub fn runAll(ctx: Ctx, tree: *const BitTree) bool {
            const total = tree.total_bits;
            if (total == 0) return true;
            if (tree.levels.items.len == 0) return true;
            const leaves = tree.levels.items[0].state.items;
            var w: usize = 0;
            while (w < leaves.len) : (w += 1) {
                const valid: u64 = if (w + 1 == leaves.len) lastLeafMask(total) else all_ones;
                if (!processLeafWord(ctx, leaves, w, valid)) return false;
            }
            return true;
        }
    };
}
pub const FlatBitSet = struct {
    const Self = @This();

    words: ListA64(u64) = .empty,
    total_bits: u32 = 0,
    active_count: u32 = 0,

    pub const empty: Self = .{};

    pub fn deinit(self: *Self, allocator: Allocator) void {
        self.words.deinit(allocator);
        self.* = undefined;
    }

    pub inline fn totalBitsCount(self: *const Self) u32 {
        return self.total_bits;
    }

    pub inline fn count(self: *const Self, state: BitState) u32 {
        return switch (state) {
            .active => self.active_count,
            .inactive => self.total_bits - self.active_count,
        };
    }

    pub fn getBit(self: *const Self, id: u32) bool {
        std.debug.assert(id < self.total_bits);
        const w: u64 = self.words.items[id >> 6];
        const s: u6 = @intCast(id & 63);
        return ((w >> s) & 1) == 1;
    }

    pub fn get(self: *const Self, bit: u32) BitState {
        return if (self.getBit(bit)) .active else .inactive;
    }

    pub fn setBit(self: *Self, id: u32, value: bool) void {
        std.debug.assert(id < self.total_bits);
        const word_idx: usize = id >> 6;
        const shift: u6 = @intCast(id & 63);
        const mask: u64 = @as(u64, 1) << shift;
        const word: *u64 = &self.words.items[word_idx];
        const was = (word.* & mask) != 0;
        if (was == value) return;
        if (value) {
            word.* |= mask;
            self.active_count += 1;
        } else {
            word.* &= ~mask;
            self.active_count -= 1;
        }
    }

    pub fn set(self: *Self, bit: u32, state: BitState) void {
        self.setBit(bit, state == .active);
    }

    inline fn wordsFor(bits: u32) usize {
        return std.math.divCeil(u32, bits, 64) catch unreachable;
    }

    inline fn lastWordMask(total_bits: u32) u64 {
        const r: u6 = @intCast(total_bits & 63);
        if (r == 0) return all_ones;
        return (@as(u64, 1) << r) - 1;
    }

    pub fn resize(self: *Self, allocator: Allocator, new_bits_count: u32, new_bits_state: BitState) !void {
        const old_bits = self.total_bits;
        if (new_bits_count == old_bits) return;
        const old_words: usize = self.words.items.len;

        var removed_active: u32 = 0;
        if (new_bits_count < old_bits and old_words > 0) {
            removed_active = self.countRangeActive(new_bits_count, old_bits - new_bits_count);
        }

        const need_words: usize = wordsFor(new_bits_count);
        try self.words.resize(allocator, need_words);
        self.total_bits = new_bits_count;

        var added_active: u32 = 0;
        if (new_bits_count > old_bits) {
            if (new_bits_state == .active) {
                if (old_bits > 0 and old_bits & 63 != 0) {
                    const tail_idx: usize = old_words - 1;
                    const word_start: u64 = @as(u64, tail_idx) * 64;
                    const fill_to: u64 = @min(@as(u64, new_bits_count), word_start + 64);
                    const lo: u32 = @intCast(@as(u64, old_bits) - word_start);
                    const hi: u32 = @intCast(fill_to - word_start);
                    var m: u64 = all_ones << @as(u6, @intCast(lo));
                    if (hi < 64) {
                        m &= (@as(u64, 1) << @as(u6, @intCast(hi))) - 1;
                    }
                    self.words.items[tail_idx] |= m;
                }
                if (old_words < self.words.items.len) {
                    @memset(self.words.items[old_words..], all_ones);
                }
                added_active = new_bits_count - old_bits;
            } else {
                if (old_words < self.words.items.len) {
                    @memset(self.words.items[old_words..], 0);
                }
            }
            if (new_bits_count & 63 != 0) {
                self.words.items[self.words.items.len - 1] &= lastWordMask(new_bits_count);
            }
        } else {
            if (new_bits_count > 0 and new_bits_count & 63 != 0) {
                self.words.items[self.words.items.len - 1] &= lastWordMask(new_bits_count);
            }
        }
        self.active_count = self.active_count - removed_active + added_active;
    }

    fn countRangeActive(self: *const Self, start: u32, len: u32) u32 {
        if (len == 0) return 0;
        const end: u64 = @as(u64, start) + len;
        var w: usize = @intCast(start >> 6);
        const last_word: usize = @intCast((end - 1) >> 6);
        var acc: u32 = 0;
        while (w <= last_word) : (w += 1) {
            const ws: u64 = @as(u64, w) * 64;
            var m: u64 = all_ones;
            if (@as(u64, start) > ws) {
                const lo: u6 = @intCast(@as(u64, start) - ws);
                m &= all_ones << lo;
            }
            if (end - ws < 64) {
                const hi: u6 = @intCast(end - ws);
                m &= (@as(u64, 1) << hi) - 1;
            }
            acc += @intCast(@popCount(self.words.items[w] & m));
        }
        return acc;
    }
};

// ---------------- tests ----------------

// Shared collection context for callback tests: appends ids, bounds-checked.
const IdCollector = struct {
    buf: []u32,
    n: usize = 0,
    fn push(self: *IdCollector, id: u32) bool {
        if (self.n >= self.buf.len) return false;
        self.buf[self.n] = id;
        self.n += 1;
        return true;
    }
};

// Three-way collection context for single-layer tests.
const TriCollector = struct {
    a_buf: []u32,
    i_buf: []u32,
    m_buf: []u32,
    na: usize = 0,
    ni: usize = 0,
    nm: usize = 0,
    fn pushA(self: *TriCollector, id: u32) bool {
        if (self.na >= self.a_buf.len) return false;
        self.a_buf[self.na] = id;
        self.na += 1;
        return true;
    }
    fn pushI(self: *TriCollector, id: u32) bool {
        if (self.ni >= self.i_buf.len) return false;
        self.i_buf[self.ni] = id;
        self.ni += 1;
        return true;
    }
    fn pushM(self: *TriCollector, id: u32) bool {
        if (self.nm >= self.m_buf.len) return false;
        self.m_buf[self.nm] = id;
        self.nm += 1;
        return true;
    }
    fn pushMt(self: *TriCollector, tree: *const BitTree, id: u32) bool {
        _ = tree;
        return self.pushM(id);
    }
};

// Early-break context: stops after exactly `limit` callbacks.
const BreakCtx = struct {
    calls: usize = 0,
    limit: usize,
    fn cb(self: *BreakCtx, id: u32) bool {
        _ = id;
        self.calls += 1;
        return self.calls < self.limit;
    }
};

test "empty tree iterator apis" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try std.testing.expectEqual(@as(u32, 0), tree.totalBitsCount());
    try std.testing.expectEqual(@as(u32, 0), tree.count(.active));
    var buf: [4]u32 = undefined;
    var coll = IdCollector{ .buf = buf[0..] };
    try std.testing.expect(tree.iterateTargetBits(*IdCollector, &coll, IdCollector.push, IdCollector.push));
    try std.testing.expectEqual(@as(usize, 0), coll.n);
    const Leaf = BitsetIterator(*IdCollector, 0, IdCollector.push, null);
    try std.testing.expect(Leaf.runAll(&coll, &tree));
    try std.testing.expectEqual(@as(usize, 0), coll.n);
}

test "setBit/getBit small" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 100, .inactive);
    tree.setBit(3, true);
    tree.setBit(5, true);
    tree.setBit(64, true);
    tree.setBit(65, true);
    try std.testing.expect(tree.getBit(3));
    try std.testing.expect(!tree.getBit(4));
    try std.testing.expectEqual(BitState.active, tree.get(65));
    try std.testing.expectEqual(@as(u32, 4), tree.count(.active));
    tree.set(5, .active);
    try std.testing.expectEqual(@as(u32, 4), tree.count(.active));
    tree.setBit(3, false);
    try std.testing.expectEqual(@as(u32, 3), tree.count(.active));
}

test "resize precise grow/shrink dual mask" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 150, .inactive);
    try std.testing.expectEqual(@as(u32, 150), tree.totalBitsCount());
    tree.set(10, .active);
    tree.set(149, .active);
    try tree.resize(alloc, 200, .active);
    try std.testing.expect(tree.getBit(10));
    try std.testing.expect(tree.getBit(149));
    try std.testing.expect(tree.getBit(150));
    try std.testing.expect(tree.getBit(199));
    try std.testing.expectEqual(@as(u32, 2 + 50), tree.count(.active));
    try tree.resize(alloc, 11, .inactive);
    try std.testing.expectEqual(@as(u32, 11), tree.totalBitsCount());
    try std.testing.expectEqual(@as(u32, 1), tree.count(.active));
    try std.testing.expect(tree.getBit(10));
    try tree.resize(alloc, 0, .inactive);
    try std.testing.expectEqual(@as(u32, 0), tree.totalBitsCount());
    try tree.resize(alloc, 70, .active);
    try std.testing.expectEqual(@as(u32, 70), tree.count(.active));
}

test "setRange and clear dual mask" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 300, .inactive);
    tree.setRange(60, 10, .active);
    try std.testing.expectEqual(@as(u32, 10), tree.count(.active));
    tree.setRange(0, 300, .active);
    try std.testing.expectEqual(@as(u32, 300), tree.count(.active));
    tree.setRange(100, 100, .inactive);
    try std.testing.expectEqual(@as(u32, 200), tree.count(.active));
    try std.testing.expect(tree.getBit(99));
    try std.testing.expect(!tree.getBit(100));
    try std.testing.expect(!tree.getBit(199));
    try std.testing.expect(tree.getBit(200));
    tree.clear(.inactive);
    try std.testing.expectEqual(@as(u32, 0), tree.count(.active));
    tree.clear(.active);
    try std.testing.expectEqual(@as(u32, 300), tree.count(.active));
}

test "layer iterator single stage exact order" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 5000, .inactive);
    tree.setRange(1000, 3000, .active);
    // L1 has ceil(5000/64) = 79 ids in 2 words.
    var a_buf: [256]u32 = undefined;
    var i_buf: [256]u32 = undefined;
    var m_buf: [256]u32 = undefined;
    var t = TriCollector{ .a_buf = a_buf[0..], .i_buf = i_buf[0..], .m_buf = m_buf[0..] };
    const L1 = LayerBitsIterator(*TriCollector, 1, 1, .{ .active = 1, .inactive = 1, .mixed = 1 }, TriCollector.pushA, TriCollector.pushI, TriCollector.pushMt);
    try std.testing.expect(L1.runAll(&t, &tree));
    try std.testing.expectEqual(@as(usize, 79), t.na + t.ni + t.nm);
    try std.testing.expect(t.na > 0 and t.ni > 0 and t.nm > 0);
    // Exact ascending partition against a naive per-span classification.
    var ia: usize = 0;
    var ii: usize = 0;
    var im: usize = 0;
    var id: u32 = 0;
    while (id < 79) : (id += 1) {
        var saw_a = false;
        var saw_i = false;
        var b: u64 = @as(u64, id) * 64;
        const end: u64 = @min(b + 64, 5000);
        while (b < end) : (b += 1) {
            if (tree.getBit(@intCast(b))) {
                saw_a = true;
            } else {
                saw_i = true;
            }
        }
        if (saw_a and saw_i) {
            try std.testing.expect(im < t.nm);
            try std.testing.expectEqual(id, t.m_buf[im]);
            im += 1;
        } else if (saw_a) {
            try std.testing.expect(ia < t.na);
            try std.testing.expectEqual(id, t.a_buf[ia]);
            ia += 1;
        } else {
            try std.testing.expect(ii < t.ni);
            try std.testing.expectEqual(id, t.i_buf[ii]);
            ii += 1;
        }
    }
    try std.testing.expectEqual(t.na, ia);
    try std.testing.expectEqual(t.ni, ii);
    try std.testing.expectEqual(t.nm, im);
}

test "manual chain L2-L1-leaf exact bits" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 20000, .inactive);
    tree.setRange(0, 5000, .active);
    var buf: [5000]u32 = undefined;
    var coll = IdCollector{ .buf = buf[0..] };
    const Leaf = BitsetIterator(*IdCollector, 1, IdCollector.push, null);
    const L1 = LayerBitsIterator(*IdCollector, 1, 2, .{ .active = 0, .inactive = 0, .mixed = 1 }, IdCollector.push, null, Leaf.step);
    const L2 = LayerBitsIterator(*IdCollector, 2, 2, .{ .active = 0, .inactive = 0, .mixed = 2 }, IdCollector.push, null, L1.step);
    try std.testing.expect(L2.runAll(&coll, &tree));
    // Active set is exactly [0, 5000), ascending by depth-first order.
    try std.testing.expectEqual(@as(usize, 5000), coll.n);
    var k: usize = 0;
    while (k < coll.n) : (k += 1) {
        try std.testing.expectEqual(@as(u32, @intCast(k)), coll.buf[k]);
    }
    // Restricted step: L2 id 1 covers bits [4096, 8192); actives are [4096, 5000).
    var coll2 = IdCollector{ .buf = buf[0..] };
    try std.testing.expect(L2.step(&coll2, &tree, 1));
    try std.testing.expectEqual(@as(usize, 904), coll2.n);
    var j: usize = 0;
    while (j < coll2.n) : (j += 1) {
        try std.testing.expectEqual(@as(u32, @intCast(4096 + j)), coll2.buf[j]);
    }
    // Out-of-range step is a silent no-op.
    var coll3 = IdCollector{ .buf = buf[0..] };
    try std.testing.expect(L2.step(&coll3, &tree, 9999));
    try std.testing.expectEqual(@as(usize, 0), coll3.n);
}

test "OutLayers split granularities" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 20000, .inactive);
    tree.setRange(0, 5000, .active);
    // Active as bits (0), inactive as L1 ids (1), mixed as L2 ids (2).
    var a_buf: [5000]u32 = undefined;
    var i_buf: [313]u32 = undefined;
    var m_buf: [16]u32 = undefined;
    var t = TriCollector{ .a_buf = a_buf[0..], .i_buf = i_buf[0..], .m_buf = m_buf[0..] };
    const S = LayerBitsIterator(*TriCollector, 2, 2, .{ .active = 0, .inactive = 1, .mixed = 2 }, TriCollector.pushA, TriCollector.pushI, TriCollector.pushMt);
    try std.testing.expect(S.runAll(&t, &tree));
    // Only uniform-active regions reach on_active: L2 slot 0 = bits [0, 4096).
    // Actives inside mixed slot 1 ([4096, 5000)) go to on_mixed as L2 id 1.
    try std.testing.expectEqual(@as(usize, 4096), t.na);
    var k: usize = 0;
    while (k < t.na) : (k += 1) {
        try std.testing.expectEqual(@as(u32, @intCast(k)), t.a_buf[k]);
    }
    // L2 has ceil(20000/4096) = 5 ids: [0]=active, [1]=mixed, [2..4]=inactive.
    try std.testing.expectEqual(@as(usize, 1), t.nm);
    try std.testing.expectEqual(@as(u32, 1), t.m_buf[0]);
    // L1 has ceil(20000/64) = 313 ids; uniform-inactive L2 slots are 2,3,4,
    // i.e. L1 ids [128, 313). Inactive ids inside mixed L2 slot 1 are not
    // visited by this stage at all (slot 1 goes to on_mixed as a whole).
    try std.testing.expectEqual(@as(usize, 185), t.ni);
    var j: usize = 0;
    while (j < t.ni) : (j += 1) {
        try std.testing.expectEqual(@as(u32, @intCast(128 + j)), t.i_buf[j]);
    }
}

test "early break stops walk" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 5000, .active);
    var b = BreakCtx{ .limit = 100 };
    try std.testing.expect(!tree.iterateTargetBits(*BreakCtx, &b, BreakCtx.cb, null));
    try std.testing.expectEqual(@as(usize, 100), b.calls);
    var b2 = BreakCtx{ .limit = std.math.maxInt(usize) };
    try std.testing.expect(tree.iterateTargetBits(*BreakCtx, &b2, BreakCtx.cb, null));
    try std.testing.expectEqual(@as(usize, 5000), b2.calls);
}

test "iterateTargetBits exact order" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 5000, .inactive);
    tree.setRange(1000, 3000, .active);
    tree.setBit(0, true);
    tree.setBit(4999, true);
    var buf: [5000]u32 = undefined;
    var coll = IdCollector{ .buf = buf[0..] };
    try std.testing.expect(tree.iterateTargetBits(*IdCollector, &coll, IdCollector.push, null));
    try std.testing.expectEqual(@as(usize, 3002), coll.n);
    // Strictly ascending and exactly the naive active set.
    var k: usize = 0;
    var b: u32 = 0;
    while (b < 5000) : (b += 1) {
        const want = (b >= 1000 and b < 4000) or b == 0 or b == 4999;
        if (!want) continue;
        try std.testing.expect(k < coll.n);
        try std.testing.expectEqual(b, coll.buf[k]);
        k += 1;
    }
    try std.testing.expectEqual(coll.n, k);
    var buf_i: [5000]u32 = undefined;
    var coll_i = IdCollector{ .buf = buf_i[0..] };
    try std.testing.expect(tree.iterateTargetBits(*IdCollector, &coll_i, null, IdCollector.push));
    try std.testing.expectEqual(@as(usize, 1998), coll_i.n);
    var k2: usize = 0;
    var b2: u32 = 0;
    while (b2 < 5000) : (b2 += 1) {
        const want = (b2 >= 1000 and b2 < 4000) or b2 == 0 or b2 == 4999;
        if (want) continue;
        try std.testing.expect(k2 < coll_i.n);
        try std.testing.expectEqual(b2, coll_i.buf[k2]);
        k2 += 1;
    }
    try std.testing.expectEqual(coll_i.n, k2);
}

test "guards: empty, shallow, out-of-range" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    var buf: [8]u32 = undefined;
    var coll = IdCollector{ .buf = buf[0..] };
    try std.testing.expect(tree.iterateTargetBits(*IdCollector, &coll, IdCollector.push, IdCollector.push));
    try std.testing.expectEqual(@as(usize, 0), coll.n);
    try tree.resize(alloc, 100, .inactive);
    // scan_layer 5 does not exist in a depth-2 tree: silent no-op.
    var coll2 = IdCollector{ .buf = buf[0..] };
    const Deep2 = LayerBitsIterator(*IdCollector, 5, 5, .{ .active = 5, .inactive = 0, .mixed = 0 }, IdCollector.push, null, null);
    try std.testing.expect(Deep2.runAll(&coll2, &tree));
    try std.testing.expectEqual(@as(usize, 0), coll2.n);
    // Out-of-range step id is a silent no-op (L1 has 2 ids here).
    const L1 = LayerBitsIterator(*IdCollector, 1, 1, .{ .active = 1, .inactive = 1, .mixed = 1 }, IdCollector.push, IdCollector.push, null);
    var coll3 = IdCollector{ .buf = buf[0..] };
    try std.testing.expect(L1.step(&coll3, &tree, 99));
    try std.testing.expectEqual(@as(usize, 0), coll3.n);
    const Leaf = BitsetIterator(*IdCollector, 0, IdCollector.push, null);
    var coll4 = IdCollector{ .buf = buf[0..] };
    try std.testing.expect(Leaf.step(&coll4, &tree, 1000));
    try std.testing.expectEqual(@as(usize, 0), coll4.n);
}

test "fuzz target vs naive model" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    var model: [1024]bool = [_]bool{false} ** 1024;
    var model_len: u32 = 0;
    var seed: u64 = 0x1234_5678_9ABC_DEF1;
    const rnd = struct {
        fn next(s: *u64) u64 {
            var x = s.*;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            s.* = x;
            return x;
        }
    }.next;
    var step: u32 = 0;
    while (step < 500) : (step += 1) {
        const op = rnd(&seed) % 100;
        if (op < 40 and model_len > 0) {
            const b: u32 = @intCast(rnd(&seed) % model_len);
            const v: bool = rnd(&seed) % 2 == 0;
            tree.setBit(b, v);
            model[b] = v;
        } else if (op < 55 and model_len > 0) {
            const s: u32 = @intCast(rnd(&seed) % model_len);
            const max_l: u32 = @min(model_len - s, 200);
            const ln: u32 = @intCast(rnd(&seed) % (max_l + 1));
            const v: bool = rnd(&seed) % 2 == 0;
            tree.setRange(s, ln, if (v) .active else .inactive);
            var i: u32 = 0;
            while (i < ln) : (i += 1) model[s + i] = v;
        } else if (op < 65) {
            const new_len: u32 = @intCast(rnd(&seed) % 1025);
            const v: bool = rnd(&seed) % 2 == 0;
            try tree.resize(alloc, new_len, if (v) .active else .inactive);
            if (new_len > model_len) {
                var i: u32 = model_len;
                while (i < new_len) : (i += 1) model[i] = v;
            }
            model_len = new_len;
        }
        // Verify counts and spot getBit.
        var c: u32 = 0;
        var i: u32 = 0;
        while (i < model_len) : (i += 1) {
            if (model[i]) {
                c += 1;
            }
        }
        try std.testing.expectEqual(c, tree.count(.active));
        if (model_len > 0) {
            const b: u32 = @intCast(rnd(&seed) % model_len);
            try std.testing.expectEqual(model[b], tree.getBit(b));
        }
        // Periodically verify full target scan.
        if (step % 50 == 0) {
            var buf: [1024]u32 = undefined;
            var coll = IdCollector{ .buf = buf[0..] };
            try std.testing.expect(tree.iterateTargetBits(*IdCollector, &coll, IdCollector.push, null));
            const n: usize = coll.n;
            try std.testing.expectEqual(@as(usize, c), n);
            var seen = [_]bool{false} ** 1024;
            var k: usize = 0;
            while (k < n) : (k += 1) {
                try std.testing.expect(buf[k] < model_len);
                try std.testing.expect(model[buf[k]]);
                try std.testing.expect(!seen[buf[k]]);
                seen[buf[k]] = true;
            }
            var j: u32 = 0;
            while (j < model_len) : (j += 1) {
                if (model[j]) {
                    try std.testing.expect(seen[j]);
                }
            }
        }
    }
}

test "flat getBit/setBit" {
    const alloc = std.testing.allocator;
    var bits = FlatBitSet.empty;
    defer bits.deinit(alloc);
    try bits.resize(alloc, 100, .inactive);
    bits.setBit(3, true);
    bits.setBit(65, true);
    try std.testing.expect(bits.getBit(3));
    try std.testing.expect(!bits.getBit(4));
    try std.testing.expectEqual(@as(u32, 2), bits.count(.active));
    bits.set(3, .inactive);
    try std.testing.expectEqual(@as(u32, 1), bits.count(.active));
}
