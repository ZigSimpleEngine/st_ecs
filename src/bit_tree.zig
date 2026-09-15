/// BitTree with dual-mask hierarchy for layer-by-layer slice generation.
/// Level 0 holds raw bits (64 per u64). Levels >=1 hold pairs:
/// state word + mixed word. Encoding per slot: (0,0)=uniform inactive,
/// (1,0)=uniform active, (0,1)=shallow mixed, (1,1)=deep mixed
/// (whole subtree fragmented down to L0, safe to jump straight to leaves).
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
/// Divisor is a power of two: shift instead of division.
inline fn leafWordsFor(bits: u32) usize {
    return (@as(usize, bits) + leaf_fanout - 1) >> 6;
}

/// Counts parent summary words needed to cover the given child word count.
/// Divisor is a power of two: shift instead of division.
inline fn parentWordsFor(child_words: usize) usize {
    return (child_words + node_fanout - 1) >> 6;
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

/// Total ids at the given layer: ceil(total_bits / span).
/// Span is always a power of two (64^layer): shift instead of division.
inline fn totalIdsForLayer(total_bits: u32, layer: u32) u64 {
    // Every caller proves total_bits != 0 first (stepSlice/runAll early-return,
    // non-empty levels imply total > 0): the branch was never taken on hot paths,
    // so it is a Debug-only contract now and free in release.
    std.debug.assert(total_bits != 0);
    const sh: u6 = @intCast(6 * layer);
    return (@as(u64, total_bits) + (@as(u64, 1) << sh) - 1) >> sh;
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

/// Slice of consecutive ids in one hierarchy layer: [start, end).
/// Polarity is known statically from which callback fired:
/// on_active = uniform active (1,0), on_inactive = uniform inactive (0,0),
/// on_mixed = shallow mixed (0,1), on_deep_mixed = deep mixed (1,1).
/// No state/mixed bits are carried: the receiver knows them from the callback.
/// The iterator always emits in its own scan layer; the receiver converts the
/// range to whatever layer it needs via `bitBase` / `bitLen`.
pub const BitsSlice = struct {
    /// Inclusive start id in units of `layer`.
    start: u32,
    /// Exclusive end id in units of `layer`.
    end: u32,
    /// Hierarchy layer of the ids (0 = raw bits). Plain u8 for cheap shifts.
    layer: u8,

    /// Number of covered ids in slice units.
    pub inline fn len(self: BitsSlice) u64 {
        return @as(u64, self.end) - self.start;
    }

    /// First covered raw bit id.
    pub inline fn bitBase(self: BitsSlice) u64 {
        std.debug.assert(self.layer < max_levels);
        return @as(u64, self.start) << @as(u6, @intCast(@as(u32, self.layer) * 6));
    }

    /// Count of covered raw bits.
    pub inline fn bitLen(self: BitsSlice) u64 {
        std.debug.assert(self.layer < max_levels);
        return self.len() << @as(u6, @intCast(@as(u32, self.layer) * 6));
    }
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
    /// Encoding: (0,0)=uniform inactive, (1,0)=uniform active,
    /// (0,1)=shallow mixed, (1,1)=deep mixed (fully fragmented).
    /// L1 mixed slots are always deep: below them are only raw bits,
    /// so there is no intermediate layer left to skip.
    const SlotSummary = struct { s: bool, m: bool };
    inline fn summarizeLeafBits(word: u64, valid_mask: u64) SlotSummary {
        const w = word & valid_mask;
        if (w == 0) return .{ .s = false, .m = false };
        if (w == valid_mask) return .{ .s = true, .m = false };
        return .{ .s = true, .m = true };
    }

    /// Summarizes one child summary word (pair) into (state_bit, mixed_bit).
    /// Child covers up to 64 slots, only `valid` of them are real.
    /// Deep mixed (1,1) means every valid child slot is mixed and itself
    /// deep, i.e. the whole subtree holds no uniform node down to L0.
    inline fn summarizeChildWord(child_state: u64, child_mixed: u64, valid: u32) SlotSummary {
        const vm = slotsValidMask(valid);
        const m = child_mixed & vm;
        const s = child_state & vm;
        if (m == 0) {
            if (s == 0) return .{ .s = false, .m = false };
            if (s == vm) return .{ .s = true, .m = false };
            return .{ .s = false, .m = true };
        }
        if (m == vm and s == vm) return .{ .s = true, .m = true };
        return .{ .s = false, .m = true };
    }

    /// Writes one slot bit pair into parent words. Returns true when changed.
    inline fn writeSlot(parent_state: *u64, parent_mixed: *u64, slot: u32, s: bool, m: bool) bool {
        const slot6: u6 = @intCast(slot);
        const bit: u64 = @as(u64, 1) << slot6;
        const old_s = (parent_state.* & bit) != 0;
        const old_m = (parent_mixed.* & bit) != 0;
        if (old_s == s and old_m == m) return false;
        // Branchless insert (early-exit above stays: it avoids dirtying the
        // cache line and stops propagation, a branchless write would lose that).
        const s_bit: u64 = @as(u64, @intFromBool(s)) << slot6;
        const m_bit: u64 = @as(u64, @intFromBool(m)) << slot6;
        parent_state.* = (parent_state.* & ~bit) | s_bit;
        parent_mixed.* = (parent_mixed.* & ~bit) | m_bit;
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
        // Branchless select, loop-invariant: splat is all-ones for fill-1 and
        // zero for fill-0, so new = (old & ~m) | (m & splat) covers both.
        const splat: u64 = @as(u64, 0) -% @as(u64, @intFromBool(want_one));
        var changed: u32 = 0;
        // Steady-split: edge words carry partial masks, middle full words need
        // no rangeMask call at all (mask is all_ones, so new is splat).
        if (first_word == last_word) {
            const m = rangeMask(first_word, start, end);
            const old = leaves[first_word];
            const new = (old & ~m) | (m & splat);
            if (new != old) {
                leaves[first_word] = new;
                // Bits flip in exactly one direction per call: single popCount.
                const flipped: u32 = @intCast(@popCount(old ^ new));
                if (want_one) {
                    self.active_count += flipped;
                } else {
                    self.active_count -= flipped;
                }
                changed += 1;
            }
        } else {
            // First partial word: bits [start&63, 64).
            {
                const m: u64 = all_ones << @as(u6, @intCast(start & 63));
                const old = leaves[first_word];
                const new = (old & ~m) | (m & splat);
                if (new != old) {
                    leaves[first_word] = new;
                    const flipped: u32 = @intCast(@popCount(old ^ new));
                    if (want_one) {
                        self.active_count += flipped;
                    } else {
                        self.active_count -= flipped;
                    }
                    changed += 1;
                }
            }
            // Middle full words.
            var w: usize = first_word + 1;
            while (w < last_word) : (w += 1) {
                const old = leaves[w];
                if (old != splat) {
                    leaves[w] = splat;
                    const flipped: u32 = @intCast(@popCount(old ^ splat));
                    if (want_one) {
                        self.active_count += flipped;
                    } else {
                        self.active_count -= flipped;
                    }
                    changed += 1;
                }
            }
            // Last partial word: bits [0, end&63), full when end is aligned.
            {
                const end_lo: u64 = end & 63;
                const m: u64 = if (end_lo == 0) all_ones else (((@as(u64, 1) << @as(u6, @intCast(end_lo))) - 1));
                const old = leaves[last_word];
                const new = (old & ~m) | (m & splat);
                if (new != old) {
                    leaves[last_word] = new;
                    const flipped: u32 = @intCast(@popCount(old ^ new));
                    if (want_one) {
                        self.active_count += flipped;
                    } else {
                        self.active_count -= flipped;
                    }
                    changed += 1;
                }
            }
        }
        if (changed == 0) return;
        if (changed > node_fanout * 4) {
            self.rebuildSummaries();
        } else {
            var w: usize = first_word;
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
        const first_word: usize = @intCast(start >> 6);
        const last_word: usize = @intCast((end - 1) >> 6);
        var acc: u32 = 0;
        // Steady-split: single rangeMask call for edge words, bare popCount
        // for middle full words (no mask AND at all).
        if (first_word == last_word) {
            acc += @intCast(@popCount(leaves[first_word] & rangeMask(first_word, start, end)));
            return acc;
        }
        acc += @intCast(@popCount(leaves[first_word] & (all_ones << @as(u6, @intCast(start & 63)))));
        var w: usize = first_word + 1;
        while (w < last_word) : (w += 1) {
            acc += @intCast(@popCount(leaves[w]));
        }
        const end_lo: u64 = end & 63;
        const last_mask: u64 = if (end_lo == 0) all_ones else (((@as(u64, 1) << @as(u6, @intCast(end_lo))) - 1));
        acc += @intCast(@popCount(leaves[last_word] & last_mask));
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

    /// Per-bit descent built on slices (fair adapter + test oracle).
    /// Same ascending, depth-first semantics as the old stage chain: uniform
    /// slices are expanded to per-bit callbacks, mixed/deep slices descend
    /// via `stepSlice`. Deep slices route to the mixed link (external code
    /// that wants the deep shortcut uses `LayerSlicesIterator` directly).
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
        const C = TargetChain(Ctx, on_active, on_inactive);
        const c = C{ .user = ctx };
        // Deep slices bypass intermediate levels via the external leaf jump.
        // Shallow slices keep stepping down: their subtrees may hold uniform
        // nodes that expand cheaper one level at a time.
        const deepCb = if (on_active != null and on_inactive != null) C.deepBoth else if (on_active != null) C.deepActive else C.deepInactive;
        const depth: usize = self.levels.items.len;
        switch (depth) {
            0 => return true,
            1 => {
                const L0 = LayerSlicesIterator(C, 0, C.active, C.inactive, null, null);
                return L0.runAll(c, self);
            },
            2 => {
                const L0 = LayerSlicesIterator(C, 0, C.active, C.inactive, null, null);
                const L1 = LayerSlicesIterator(C, 1, C.active, C.inactive, L0.stepSlice, deepCb);
                return L1.runAll(c, self);
            },
            3 => {
                const L0 = LayerSlicesIterator(C, 0, C.active, C.inactive, null, null);
                const L1 = LayerSlicesIterator(C, 1, C.active, C.inactive, L0.stepSlice, deepCb);
                const L2 = LayerSlicesIterator(C, 2, C.active, C.inactive, L1.stepSlice, deepCb);
                return L2.runAll(c, self);
            },
            4 => {
                const L0 = LayerSlicesIterator(C, 0, C.active, C.inactive, null, null);
                const L1 = LayerSlicesIterator(C, 1, C.active, C.inactive, L0.stepSlice, deepCb);
                const L2 = LayerSlicesIterator(C, 2, C.active, C.inactive, L1.stepSlice, deepCb);
                const L3 = LayerSlicesIterator(C, 3, C.active, C.inactive, L2.stepSlice, deepCb);
                return L3.runAll(c, self);
            },
            5 => {
                const L0 = LayerSlicesIterator(C, 0, C.active, C.inactive, null, null);
                const L1 = LayerSlicesIterator(C, 1, C.active, C.inactive, L0.stepSlice, deepCb);
                const L2 = LayerSlicesIterator(C, 2, C.active, C.inactive, L1.stepSlice, deepCb);
                const L3 = LayerSlicesIterator(C, 3, C.active, C.inactive, L2.stepSlice, deepCb);
                const L4 = LayerSlicesIterator(C, 4, C.active, C.inactive, L3.stepSlice, deepCb);
                return L4.runAll(c, self);
            },
            6 => {
                const L0 = LayerSlicesIterator(C, 0, C.active, C.inactive, null, null);
                const L1 = LayerSlicesIterator(C, 1, C.active, C.inactive, L0.stepSlice, deepCb);
                const L2 = LayerSlicesIterator(C, 2, C.active, C.inactive, L1.stepSlice, deepCb);
                const L3 = LayerSlicesIterator(C, 3, C.active, C.inactive, L2.stepSlice, deepCb);
                const L4 = LayerSlicesIterator(C, 4, C.active, C.inactive, L3.stepSlice, deepCb);
                const L5 = LayerSlicesIterator(C, 5, C.active, C.inactive, L4.stepSlice, deepCb);
                return L5.runAll(c, self);
            },
            7 => {
                const L0 = LayerSlicesIterator(C, 0, C.active, C.inactive, null, null);
                const L1 = LayerSlicesIterator(C, 1, C.active, C.inactive, L0.stepSlice, deepCb);
                const L2 = LayerSlicesIterator(C, 2, C.active, C.inactive, L1.stepSlice, deepCb);
                const L3 = LayerSlicesIterator(C, 3, C.active, C.inactive, L2.stepSlice, deepCb);
                const L4 = LayerSlicesIterator(C, 4, C.active, C.inactive, L3.stepSlice, deepCb);
                const L5 = LayerSlicesIterator(C, 5, C.active, C.inactive, L4.stepSlice, deepCb);
                const L6 = LayerSlicesIterator(C, 6, C.active, C.inactive, L5.stepSlice, deepCb);
                return L6.runAll(c, self);
            },
            8 => {
                const L0 = LayerSlicesIterator(C, 0, C.active, C.inactive, null, null);
                const L1 = LayerSlicesIterator(C, 1, C.active, C.inactive, L0.stepSlice, deepCb);
                const L2 = LayerSlicesIterator(C, 2, C.active, C.inactive, L1.stepSlice, deepCb);
                const L3 = LayerSlicesIterator(C, 3, C.active, C.inactive, L2.stepSlice, deepCb);
                const L4 = LayerSlicesIterator(C, 4, C.active, C.inactive, L3.stepSlice, deepCb);
                const L5 = LayerSlicesIterator(C, 5, C.active, C.inactive, L4.stepSlice, deepCb);
                const L6 = LayerSlicesIterator(C, 6, C.active, C.inactive, L5.stepSlice, deepCb);
                const L7 = LayerSlicesIterator(C, 7, C.active, C.inactive, L6.stepSlice, deepCb);
                return L7.runAll(c, self);
            },
            else => unreachable,
        }
    }

    /// O(slices) arithmetic sum of active bit ids: no per-bit loop.
    /// Uniform active slices contribute n*(first+last)/2 each; mixed slices
    /// descend via `stepSlice`. Demonstrates the slice ceiling for dense data.
    pub fn sumActiveIds(self: *const Self) struct { sum: u64, count: u64 } {
        var st = ArithState{};
        const c = ArithChain{ .st = &st };
        const depth: usize = self.levels.items.len;
        switch (depth) {
            0 => {},
            1 => {
                const L0 = LayerSlicesIterator(ArithChain, 0, ArithChain.active, null, null, null);
                _ = L0.runAll(c, self);
            },
            2 => {
                const L0 = LayerSlicesIterator(ArithChain, 0, ArithChain.active, null, null, null);
                const L1 = LayerSlicesIterator(ArithChain, 1, ArithChain.active, null, L0.stepSlice, ArithChain.deep);
                _ = L1.runAll(c, self);
            },
            3 => {
                const L0 = LayerSlicesIterator(ArithChain, 0, ArithChain.active, null, null, null);
                const L1 = LayerSlicesIterator(ArithChain, 1, ArithChain.active, null, L0.stepSlice, ArithChain.deep);
                const L2 = LayerSlicesIterator(ArithChain, 2, ArithChain.active, null, L1.stepSlice, ArithChain.deep);
                _ = L2.runAll(c, self);
            },
            4 => {
                const L0 = LayerSlicesIterator(ArithChain, 0, ArithChain.active, null, null, null);
                const L1 = LayerSlicesIterator(ArithChain, 1, ArithChain.active, null, L0.stepSlice, ArithChain.deep);
                const L2 = LayerSlicesIterator(ArithChain, 2, ArithChain.active, null, L1.stepSlice, ArithChain.deep);
                const L3 = LayerSlicesIterator(ArithChain, 3, ArithChain.active, null, L2.stepSlice, ArithChain.deep);
                _ = L3.runAll(c, self);
            },
            5 => {
                const L0 = LayerSlicesIterator(ArithChain, 0, ArithChain.active, null, null, null);
                const L1 = LayerSlicesIterator(ArithChain, 1, ArithChain.active, null, L0.stepSlice, ArithChain.deep);
                const L2 = LayerSlicesIterator(ArithChain, 2, ArithChain.active, null, L1.stepSlice, ArithChain.deep);
                const L3 = LayerSlicesIterator(ArithChain, 3, ArithChain.active, null, L2.stepSlice, ArithChain.deep);
                const L4 = LayerSlicesIterator(ArithChain, 4, ArithChain.active, null, L3.stepSlice, ArithChain.deep);
                _ = L4.runAll(c, self);
            },
            6 => {
                const L0 = LayerSlicesIterator(ArithChain, 0, ArithChain.active, null, null, null);
                const L1 = LayerSlicesIterator(ArithChain, 1, ArithChain.active, null, L0.stepSlice, ArithChain.deep);
                const L2 = LayerSlicesIterator(ArithChain, 2, ArithChain.active, null, L1.stepSlice, ArithChain.deep);
                const L3 = LayerSlicesIterator(ArithChain, 3, ArithChain.active, null, L2.stepSlice, ArithChain.deep);
                const L4 = LayerSlicesIterator(ArithChain, 4, ArithChain.active, null, L3.stepSlice, ArithChain.deep);
                const L5 = LayerSlicesIterator(ArithChain, 5, ArithChain.active, null, L4.stepSlice, ArithChain.deep);
                _ = L5.runAll(c, self);
            },
            7 => {
                const L0 = LayerSlicesIterator(ArithChain, 0, ArithChain.active, null, null, null);
                const L1 = LayerSlicesIterator(ArithChain, 1, ArithChain.active, null, L0.stepSlice, ArithChain.deep);
                const L2 = LayerSlicesIterator(ArithChain, 2, ArithChain.active, null, L1.stepSlice, ArithChain.deep);
                const L3 = LayerSlicesIterator(ArithChain, 3, ArithChain.active, null, L2.stepSlice, ArithChain.deep);
                const L4 = LayerSlicesIterator(ArithChain, 4, ArithChain.active, null, L3.stepSlice, ArithChain.deep);
                const L5 = LayerSlicesIterator(ArithChain, 5, ArithChain.active, null, L4.stepSlice, ArithChain.deep);
                const L6 = LayerSlicesIterator(ArithChain, 6, ArithChain.active, null, L5.stepSlice, ArithChain.deep);
                _ = L6.runAll(c, self);
            },
            8 => {
                const L0 = LayerSlicesIterator(ArithChain, 0, ArithChain.active, null, null, null);
                const L1 = LayerSlicesIterator(ArithChain, 1, ArithChain.active, null, L0.stepSlice, ArithChain.deep);
                const L2 = LayerSlicesIterator(ArithChain, 2, ArithChain.active, null, L1.stepSlice, ArithChain.deep);
                const L3 = LayerSlicesIterator(ArithChain, 3, ArithChain.active, null, L2.stepSlice, ArithChain.deep);
                const L4 = LayerSlicesIterator(ArithChain, 4, ArithChain.active, null, L3.stepSlice, ArithChain.deep);
                const L5 = LayerSlicesIterator(ArithChain, 5, ArithChain.active, null, L4.stepSlice, ArithChain.deep);
                const L6 = LayerSlicesIterator(ArithChain, 6, ArithChain.active, null, L5.stepSlice, ArithChain.deep);
                const L7 = LayerSlicesIterator(ArithChain, 7, ArithChain.active, null, L6.stepSlice, ArithChain.deep);
                _ = L7.runAll(c, self);
            },
            else => unreachable,
        }
        return .{ .sum = st.sum, .count = st.count };
    }
};

/// Leaf word range covered by a slice at layer >= 1, clipped to the tree.
/// Shared by the external deep-jump callbacks (fair adapter + arith ceiling).
inline fn deepLeafRange(tree: *const BitTree, s: BitsSlice) struct { leaves: []const u64, base: u64, end: u64, words: u64, tail: u64 } {
    const leaves = tree.levels.items[0].state.items;
    const total = tree.total_bits;
    std.debug.assert(s.layer < max_levels);
    std.debug.assert(s.layer >= 1);
    // One layer-L id spans 64^(L-1) leaf words (L0 ids are bits, not words).
    const sh: u6 = @intCast(@as(u32, s.layer - 1) * 6);
    const lb: u64 = @as(u64, s.start) << sh;
    const lc: u64 = (@as(u64, s.end) - @as(u64, s.start)) << sh;
    const words: u64 = (@as(u64, total) + 63) >> 6;
    var le: u64 = lb + lc;
    if (le > words) le = words;
    if (le > leaves.len) le = leaves.len;
    return .{ .leaves = leaves, .base = @min(lb, le), .end = le, .words = words, .tail = lastLeafMask(total) };
}

/// Per-bit expansion chain context for `BitTree.iterateTargetBits`.
/// Uniform slices are expanded to ascending bit ids with tail clipping.
fn TargetChain(comptime Ctx: type, comptime on_a: ?fn (Ctx, u32) bool, comptime on_i: ?fn (Ctx, u32) bool) type {
    return struct {
        const Self = @This();
        user: Ctx,

        pub fn active(c: Self, tree: *const BitTree, s: BitsSlice) bool {
            const f = on_a orelse return true;
            return expand(tree.total_bits, c.user, f, s);
        }

        pub fn inactive(c: Self, tree: *const BitTree, s: BitsSlice) bool {
            const f = on_i orelse return true;
            return expand(tree.total_bits, c.user, f, s);
        }

        /// External deep jump, active polarity: scans leaves directly with
        /// ctz, no intermediate levels are visited. Order is preserved: the
        /// slice range is exact, siblings come before/after untouched.
        pub fn deepActive(c: Self, tree: *const BitTree, s: BitsSlice) bool {
            const f = on_a orelse return true;
            if (tree.levels.items.len == 0) return true;
            if (s.layer == 0) return expand(tree.total_bits, c.user, f, s);
            const r = deepLeafRange(tree, s);
            if (r.base >= r.end) return true;
            var w: u64 = r.base;
            while (w < r.end) : (w += 1) {
                const lw: u64 = r.leaves[@intCast(w)];
                const allow: u64 = if (w + 1 == r.words) r.tail else all_ones;
                var bits: u64 = lw & allow;
                while (bits != 0) {
                    const b: u32 = @ctz(bits);
                    bits &= bits - 1;
                    if (!f(c.user, @intCast(w * 64 + b))) return false;
                }
            }
            return true;
        }

        /// External deep jump, inactive polarity.
        pub fn deepInactive(c: Self, tree: *const BitTree, s: BitsSlice) bool {
            const f = on_i orelse return true;
            if (tree.levels.items.len == 0) return true;
            if (s.layer == 0) return expand(tree.total_bits, c.user, f, s);
            const r = deepLeafRange(tree, s);
            if (r.base >= r.end) return true;
            var w: u64 = r.base;
            while (w < r.end) : (w += 1) {
                const lw: u64 = r.leaves[@intCast(w)];
                const allow: u64 = if (w + 1 == r.words) r.tail else all_ones;
                var bits: u64 = ~lw & allow;
                while (bits != 0) {
                    const b: u32 = @ctz(bits);
                    bits &= bits - 1;
                    if (!f(c.user, @intCast(w * 64 + b))) return false;
                }
            }
            return true;
        }

        /// External deep jump, both polarities: per-bit classification in order.
        pub fn deepBoth(c: Self, tree: *const BitTree, s: BitsSlice) bool {
            if (on_a == null) return c.deepInactive(tree, s);
            if (on_i == null) return c.deepActive(tree, s);
            if (tree.levels.items.len == 0) return true;
            if (s.layer == 0) {
                // Unreachable in practice (L0 has no mixed words): ascending classify.
                const fa0 = on_a.?;
                const fi0 = on_i.?;
                var k: u64 = s.start;
                while (k < s.end) : (k += 1) {
                    const id: u32 = @intCast(k);
                    if (tree.getBit(id)) {
                        if (!fa0(c.user, id)) return false;
                    } else {
                        if (!fi0(c.user, id)) return false;
                    }
                }
                return true;
            }
            const fa = on_a.?;
            const fi = on_i.?;
            const r = deepLeafRange(tree, s);
            if (r.base >= r.end) return true;
            var w: u64 = r.base;
            while (w < r.end) : (w += 1) {
                const lw: u64 = r.leaves[@intCast(w)];
                const allow: u64 = if (w + 1 == r.words) r.tail else all_ones;
                var wanted: u64 = allow;
                while (wanted != 0) {
                    const b: u32 = @ctz(wanted);
                    wanted &= wanted - 1;
                    const is_a = ((lw >> @as(u6, @intCast(b))) & 1) == 1;
                    if (is_a) {
                        if (!fa(c.user, @intCast(w * 64 + b))) return false;
                    } else {
                        if (!fi(c.user, @intCast(w * 64 + b))) return false;
                    }
                }
            }
            return true;
        }

        inline fn expand(total: u32, user: Ctx, f: fn (Ctx, u32) bool, s: BitsSlice) bool {
            std.debug.assert(s.layer < max_levels);
            const sh: u6 = @intCast(@as(u32, s.layer) * 6);
            const base: u64 = @as(u64, s.start) << sh;
            var cnt: u64 = (@as(u64, s.end) - @as(u64, s.start)) << sh;
            const t: u64 = total;
            if (base >= t) return true;
            if (base + cnt > t) cnt = t - base;
            var k: u64 = 0;
            while (k < cnt) : (k += 1) {
                if (!f(user, @intCast(base + k))) return false;
            }
            return true;
        }
    };
}

/// Arithmetic chain context for `BitTree.sumActiveIds`: O(1) per slice.
/// n*(first+last)/2 is computed in u128 (intermediate exceeds u64 for big N)
/// then narrowed: the final sum of 0..2^32-1 still fits u64.
const ArithState = struct {
    sum: u64 = 0,
    count: u64 = 0,
};

const ArithChain = struct {
    st: *ArithState,

    pub fn active(c: ArithChain, tree: *const BitTree, s: BitsSlice) bool {
        std.debug.assert(s.layer < max_levels);
        const sh: u6 = @intCast(@as(u32, s.layer) * 6);
        const base: u64 = @as(u64, s.start) << sh;
        var cnt: u64 = (@as(u64, s.end) - @as(u64, s.start)) << sh;
        const t: u64 = tree.total_bits;
        if (base >= t) return true;
        if (base + cnt > t) cnt = t - base;
        if (cnt == 0) return true;
        // Tiny slices are cheaper as plain adds than a u128 division.
        if (cnt <= 16) {
            var k: u64 = 0;
            while (k < cnt) : (k += 1) {
                c.st.sum += base + k;
                c.st.count += 1;
            }
            return true;
        }
        const n: u128 = cnt;
        const first: u128 = base;
        const last: u128 = base + cnt - 1;
        c.st.sum += @intCast(n * (first + last) / 2);
        c.st.count += cnt;
        return true;
    }

    /// External deep jump for the arith ceiling: ctz accumulation straight
    /// from the leaves, no intermediate levels, no u128 division.
    pub fn deep(c: ArithChain, tree: *const BitTree, s: BitsSlice) bool {
        if (tree.levels.items.len == 0) return true;
        if (s.layer == 0) return c.active(tree, s);
        const r = deepLeafRange(tree, s);
        if (r.base >= r.end) return true;
        var w: u64 = r.base;
        while (w < r.end) : (w += 1) {
            const lw: u64 = r.leaves[@intCast(w)];
            const allow: u64 = if (w + 1 == r.words) r.tail else all_ones;
            var bits: u64 = lw & allow;
            while (bits != 0) {
                const b: u32 = @ctz(bits);
                bits &= bits - 1;
                c.st.sum += w * 64 + b;
                c.st.count += 1;
            }
        }
        return true;
    }
};

/// Callback-based single-layer slice iterator factory.
/// Callbacks take (Ctx, *const BitTree, BitsSlice) and return bool (false stops the walk).
///
/// Scans words at `scan_layer` and emits ascending, non-overlapping slices in
/// `scan_layer` units. Consecutive full uniform words are coalesced into one
/// cross-word slice; other words emit per-run slices in ascending slot order.
/// Mixed and deep slots are never descended into: the receiver gets a slice
/// and descends via `stepSlice` only if it needs to. A deep slice additionally
/// hints that the whole subtree below is fragmented, so the receiver may jump
/// straight to layer 0 instead of stepping level by level.
/// `stepSlice(slice)` processes the covered `scan_layer` range of an input
/// slice with `slice.layer >= scan_layer`; `runAll()` scans the whole layer.
/// A null callback prunes that polarity at comptime (no mask work emitted for
/// pruned uniform polarities beyond classification). When `on_deep_mixed` is
/// null, deep slices fall back to `on_mixed`.
/// Returns false on the first callback that returns false, true otherwise.
/// Never allocates. Portable: no SIMD, no BMI, only ctz + shifts.
pub fn LayerSlicesIterator(
    comptime Ctx: type,
    comptime scan_layer: u32,
    comptime on_active: ?fn (Ctx, *const BitTree, BitsSlice) bool,
    comptime on_inactive: ?fn (Ctx, *const BitTree, BitsSlice) bool,
    comptime on_mixed: ?fn (Ctx, *const BitTree, BitsSlice) bool,
    comptime on_deep_mixed: ?fn (Ctx, *const BitTree, BitsSlice) bool,
) type {
    if (on_active == null and on_inactive == null and on_mixed == null and on_deep_mixed == null)
        @compileError("at least one callback required");
    // Deep falls back to mixed so callers that do not distinguish
    // shallow/deep only set on_mixed. The reverse is not allowed: a shallow
    // slice must never reach on_deep_mixed (its subtree may hold uniform nodes).
    const on_deep: ?fn (Ctx, *const BitTree, BitsSlice) bool = if (on_deep_mixed != null) on_deep_mixed else on_mixed;
    return struct {
        inline fn emitSlice(ctx: Ctx, tree: *const BitTree, start: u64, end: u64, comptime cb: ?fn (Ctx, *const BitTree, BitsSlice) bool) bool {
            const f = cb orelse return true;
            if (start >= end) return true;
            return f(ctx, tree, .{
                .start = @intCast(start),
                .end = @intCast(end),
                .layer = @intCast(scan_layer),
            });
        }

        /// Length of the same-polarity run starting at slot `s` in `mask`.
        inline fn runLen(mask: u64, s: u32) u32 {
            const shifted: u64 = mask >> @as(u6, @intCast(s));
            var r: u32 = @ctz(~shifted);
            const max: u32 = 64 - s;
            if (r > max) r = max;
            return r;
        }

        /// Mask that clears the `[s, s+r)` run bits (for advancing `rest`).
        inline fn clearRun(s: u32, r: u32) u64 {
            const w: u64 = if (r >= 64) all_ones else ((@as(u64, 1) << @as(u6, @intCast(r))) - 1);
            return ~(w << @as(u6, @intCast(s)));
        }

        inline fn rangeMask64(lo: u64, hi: u64) u64 {
            // Caller guarantees 0 <= lo < hi <= 64. Full mask with the low
            // (64-width) bits shifted out, then positioned at lo: width==64
            // falls out as (~0 >> 0) << 0, no special case, no UB.
            const width: u64 = hi - lo;
            const w6: u6 = @intCast(64 - width);
            const l6: u6 = @intCast(lo);
            return (((~@as(u64, 0)) >> w6) << l6);
        }

        /// Processes one word's `allowed` slots. `pend_pol`/`pend_start`/`pend_end`
        /// carry an open uniform run across words: 0=none, 1=active, 2=inactive.
        /// Pending runs are always flushed before any callback of a later word,
        /// so delivery stays strictly ascending even with early stop.
        inline fn processWord(
            ctx: Ctx,
            tree: *const BitTree,
            st_items: []const u64,
            mx_items: []const u64,
            w: usize,
            allowed: u64,
            pend_pol: *u2,
            pend_start: *u64,
            pend_end: *u64,
        ) bool {
            if (allowed == 0) return true;
            const st: u64 = st_items[w];
            const mx: u64 = if (comptime scan_layer == 0) 0 else mx_items[w];
            const a_all: u64 = st & ~mx & allowed;
            const sh_all: u64 = ~st & mx & allowed;
            const dp_all: u64 = st & mx & allowed;
            const i_all: u64 = allowed & ~mx & ~st;
            const w_base: u64 = @as(u64, w) * 64;

            // Fast path: a fully uniform full word extends the pending
            // cross-word run (or opens one). Anything else flushes first.
            if (allowed == all_ones) {
                if (a_all == all_ones) {
                    if (on_active != null) {
                        if (pend_pol.* == 2) {
                            if (!emitSlice(ctx, tree, pend_start.*, pend_end.*, on_inactive)) return false;
                        }
                        if (pend_pol.* != 1) {
                            pend_pol.* = 1;
                            pend_start.* = w_base;
                        }
                        pend_end.* = w_base + 64;
                        return true;
                    }
                    // Pruned polarity still breaks the other pending run.
                    if (pend_pol.* == 2) {
                        if (!emitSlice(ctx, tree, pend_start.*, pend_end.*, on_inactive)) return false;
                        pend_pol.* = 0;
                    }
                    return true;
                }
                if (i_all == all_ones) {
                    if (on_inactive != null) {
                        if (pend_pol.* == 1) {
                            if (!emitSlice(ctx, tree, pend_start.*, pend_end.*, on_active)) return false;
                        }
                        if (pend_pol.* != 2) {
                            pend_pol.* = 2;
                            pend_start.* = w_base;
                        }
                        pend_end.* = w_base + 64;
                        return true;
                    }
                    if (pend_pol.* == 1) {
                        if (!emitSlice(ctx, tree, pend_start.*, pend_end.*, on_active)) return false;
                        pend_pol.* = 0;
                    }
                    return true;
                }
            }
            // Generic path: flush any pending run, then emit ascending runs.
            if (pend_pol.* == 1) {
                if (!emitSlice(ctx, tree, pend_start.*, pend_end.*, on_active)) return false;
                pend_pol.* = 0;
            } else if (pend_pol.* == 2) {
                if (!emitSlice(ctx, tree, pend_start.*, pend_end.*, on_inactive)) return false;
                pend_pol.* = 0;
            }
            var rest: u64 = 0;
            if (on_active != null) rest |= a_all;
            if (on_inactive != null) rest |= i_all;
            if (on_mixed != null) rest |= sh_all;
            if (on_deep != null) rest |= dp_all;
            while (rest != 0) {
                const s: u32 = @ctz(rest);
                const bitm: u64 = @as(u64, 1) << @as(u6, @intCast(s));
                if (on_mixed != null and (sh_all & bitm) != 0) {
                    const r = runLen(sh_all, s);
                    if (!emitSlice(ctx, tree, w_base + s, w_base + s + r, on_mixed)) return false;
                    rest &= clearRun(s, r);
                } else if (on_deep != null and (dp_all & bitm) != 0) {
                    const r = runLen(dp_all, s);
                    if (!emitSlice(ctx, tree, w_base + s, w_base + s + r, on_deep)) return false;
                    rest &= clearRun(s, r);
                } else if (on_active != null and (a_all & bitm) != 0) {
                    const r = runLen(a_all, s);
                    if (!emitSlice(ctx, tree, w_base + s, w_base + s + r, on_active)) return false;
                    rest &= clearRun(s, r);
                } else {
                    const r = runLen(i_all, s);
                    if (on_inactive != null) {
                        if (!emitSlice(ctx, tree, w_base + s, w_base + s + r, on_inactive)) return false;
                    }
                    rest &= clearRun(s, r);
                }
            }
            return true;
        }

        inline fn flushPending(ctx: Ctx, tree: *const BitTree, pend_pol: *u2, pend_start: *u64, pend_end: *u64) bool {
            if (pend_pol.* == 1) {
                pend_pol.* = 0;
                if (!emitSlice(ctx, tree, pend_start.*, pend_end.*, on_active)) return false;
            } else if (pend_pol.* == 2) {
                pend_pol.* = 0;
                if (!emitSlice(ctx, tree, pend_start.*, pend_end.*, on_inactive)) return false;
            }
            return true;
        }

        /// Processes the covered `scan_layer` range of an input slice.
        /// Slices with `layer < scan_layer`, empty or out-of-range slices are
        /// silent no-ops. Returns false on early callback stop.
        pub fn stepSlice(ctx: Ctx, tree: *const BitTree, slice: BitsSlice) bool {
            const total = tree.total_bits;
            if (total == 0) return true;
            if (scan_layer >= tree.levels.items.len) return true;
            const sl: u32 = slice.layer;
            if (sl >= max_levels) return true;
            if (sl < scan_layer) return true;
            if (slice.end <= slice.start) return true;
            const total_in: u64 = totalIdsForLayer(total, sl);
            if (@as(u64, slice.start) >= total_in) return true;
            var end: u64 = slice.end;
            if (end > total_in) end = total_in;
            const d: u32 = sl - scan_layer;
            const sh: u6 = @intCast(6 * d);
            const base: u64 = @as(u64, slice.start) << sh;
            var cnt: u64 = (end - @as(u64, slice.start)) << sh;
            const total_scan: u64 = totalIdsForLayer(total, scan_layer);
            if (base >= total_scan) return true;
            if (base + cnt > total_scan) cnt = total_scan - base;
            const st_items: []const u64 = tree.levels.items[scan_layer].state.items;
            const mx_items: []const u64 = tree.levels.items[scan_layer].mixed.items;
            const words: usize = st_items.len;
            const w_first: usize = @intCast(base >> 6);
            const w_last: usize = @intCast((base + cnt - 1) >> 6);
            var pend_pol: u2 = 0;
            var pend_start: u64 = 0;
            var pend_end: u64 = 0;
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
                if (!processWord(ctx, tree, st_items, mx_items, w, allowed, &pend_pol, &pend_start, &pend_end)) return false;
            }
            return flushPending(ctx, tree, &pend_pol, &pend_start, &pend_end);
        }

        /// Processes every word at `scan_layer` in ascending order.
        /// Returns false on early callback stop.
        pub fn runAll(ctx: Ctx, tree: *const BitTree) bool {
            const total = tree.total_bits;
            if (total == 0) return true;
            if (scan_layer >= tree.levels.items.len) return true;
            const total_scan: u64 = totalIdsForLayer(total, scan_layer);
            const st_items: []const u64 = tree.levels.items[scan_layer].state.items;
            const mx_items: []const u64 = tree.levels.items[scan_layer].mixed.items;
            const words: usize = st_items.len;
            var pend_pol: u2 = 0;
            var pend_start: u64 = 0;
            var pend_end: u64 = 0;
            // All words but the last are fully valid: no per-word mask math.
            var w: usize = 0;
            const full_words: usize = @intCast(total_scan >> 6);
            const steady: usize = @min(full_words, words);
            while (w < steady) : (w += 1) {
                if (!processWord(ctx, tree, st_items, mx_items, w, all_ones, &pend_pol, &pend_start, &pend_end)) return false;
            }
            if (w < words) {
                const base: u64 = @as(u64, w) * 64;
                if (base < total_scan) {
                    const rem: u64 = total_scan - base;
                    if (!processWord(ctx, tree, st_items, mx_items, w, slotsValidMask(rem), &pend_pol, &pend_start, &pend_end)) return false;
                }
            }
            return flushPending(ctx, tree, &pend_pol, &pend_start, &pend_end);
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
        const first_word: usize = @intCast(start >> 6);
        const last_word: usize = @intCast((end - 1) >> 6);
        var acc: u32 = 0;
        // Steady-split: edge masks once, middle words as bare popCounts.
        if (first_word == last_word) {
            const ws: u64 = @as(u64, first_word) * 64;
            var m: u64 = all_ones;
            if (@as(u64, start) > ws) {
                const lo: u6 = @intCast(@as(u64, start) - ws);
                m &= all_ones << lo;
            }
            if (end - ws < 64) {
                const hi: u6 = @intCast(end - ws);
                m &= ((@as(u64, 1) << hi) - 1);
            }
            acc += @intCast(@popCount(self.words.items[first_word] & m));
            return acc;
        }
        acc += @intCast(@popCount(self.words.items[first_word] & (all_ones << @as(u6, @intCast(start & 63)))));
        var w: usize = first_word + 1;
        while (w < last_word) : (w += 1) {
            acc += @intCast(@popCount(self.words.items[w]));
        }
        const end_lo: u64 = end & 63;
        const last_mask: u64 = if (end_lo == 0) all_ones else (((@as(u64, 1) << @as(u6, @intCast(end_lo))) - 1));
        acc += @intCast(@popCount(self.words.items[last_word] & last_mask));
        return acc;
    }
};

// ---------------- tests ----------------

// Shared collection context for per-bit callback tests (iterateTargetBits oracle).
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

// Slice collection context: appends whole slices, bounds-checked.
const SliceCollector = struct {
    buf: []BitsSlice,
    n: usize = 0,
    fn push(self: *SliceCollector, tree: *const BitTree, s: BitsSlice) bool {
        _ = tree;
        if (self.n >= self.buf.len) return false;
        self.buf[self.n] = s;
        self.n += 1;
        return true;
    }
};

// Four-way slice collection context, one buffer per polarity.
const QuadCollector = struct {
    a_buf: []BitsSlice,
    i_buf: []BitsSlice,
    m_buf: []BitsSlice,
    d_buf: []BitsSlice,
    na: usize = 0,
    ni: usize = 0,
    nm: usize = 0,
    nd: usize = 0,
    fn pushA(self: *QuadCollector, tree: *const BitTree, s: BitsSlice) bool {
        _ = tree;
        if (self.na >= self.a_buf.len) return false;
        self.a_buf[self.na] = s;
        self.na += 1;
        return true;
    }
    fn pushI(self: *QuadCollector, tree: *const BitTree, s: BitsSlice) bool {
        _ = tree;
        if (self.ni >= self.i_buf.len) return false;
        self.i_buf[self.ni] = s;
        self.ni += 1;
        return true;
    }
    fn pushM(self: *QuadCollector, tree: *const BitTree, s: BitsSlice) bool {
        _ = tree;
        if (self.nm >= self.m_buf.len) return false;
        self.m_buf[self.nm] = s;
        self.nm += 1;
        return true;
    }
    fn pushD(self: *QuadCollector, tree: *const BitTree, s: BitsSlice) bool {
        _ = tree;
        if (self.nd >= self.d_buf.len) return false;
        self.d_buf[self.nd] = s;
        self.nd += 1;
        return true;
    }
};

// Test-only per-bit expander for direct stepSlice chains.
const BitPush = struct {
    coll: *IdCollector,
    pub fn active(c: BitPush, tree: *const BitTree, s: BitsSlice) bool {
        _ = tree;
        const sh: u6 = @intCast(@as(u32, s.layer) * 6);
        const base: u64 = @as(u64, s.start) << sh;
        const cnt: u64 = (@as(u64, s.end) - @as(u64, s.start)) << sh;
        var k: u64 = 0;
        while (k < cnt) : (k += 1) {
            if (!c.coll.push(@intCast(base + k))) return false;
        }
        return true;
    }
};

// Early-break context over slices: stops after exactly `limit` callbacks.
const SliceBreakCtx = struct {
    calls: usize = 0,
    limit: usize,
    fn cb(self: *SliceBreakCtx, tree: *const BitTree, s: BitsSlice) bool {
        _ = tree;
        _ = s;
        self.calls += 1;
        return self.calls < self.limit;
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

// Validates ascending, non-overlapping coverage of `seen` by one slice list.
fn checkSliceList(slices: []const BitsSlice, n: usize, layer: u8, seen: []bool) !void {
    var k: usize = 0;
    var first = true;
    var prev_end: u32 = 0;
    while (k < n) : (k += 1) {
        const s = slices[k];
        try std.testing.expectEqual(layer, s.layer);
        try std.testing.expect(s.end > s.start);
        if (!first) try std.testing.expect(s.start >= prev_end);
        first = false;
        prev_end = s.end;
        var id: u32 = s.start;
        while (id < s.end) : (id += 1) {
            try std.testing.expect(id < seen.len);
            try std.testing.expect(!seen[id]);
            seen[id] = true;
        }
    }
}

// Validates slice purity against the naive bit model.
// Uniform slices must be pure, mixed/deep slices must contain both states.
fn checkSlicePurity(s: BitsSlice, expect_mixed: bool, tree: *const BitTree, total_bits: u32) !void {
    const base: u64 = s.bitBase();
    try std.testing.expect(base < total_bits);
    const blen: u64 = @min(s.bitLen(), @as(u64, total_bits) - base);
    try std.testing.expect(blen > 0);
    var saw_a = false;
    var saw_i = false;
    var b: u64 = 0;
    while (b < blen) : (b += 1) {
        if (tree.getBit(@intCast(base + b))) {
            saw_a = true;
        } else {
            saw_i = true;
        }
        if (saw_a and saw_i) break;
    }
    if (expect_mixed) {
        try std.testing.expect(saw_a and saw_i);
    } else {
        try std.testing.expect(saw_a != saw_i);
    }
}

test "slices: empty tree" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try std.testing.expectEqual(@as(u32, 0), tree.totalBitsCount());
    try std.testing.expectEqual(@as(u32, 0), tree.count(.active));
    var buf: [4]BitsSlice = undefined;
    var coll = SliceCollector{ .buf = buf[0..] };
    const L0 = LayerSlicesIterator(*SliceCollector, 0, SliceCollector.push, SliceCollector.push, SliceCollector.push, SliceCollector.push);
    try std.testing.expect(L0.runAll(&coll, &tree));
    try std.testing.expectEqual(@as(usize, 0), coll.n);
    const L1 = LayerSlicesIterator(*SliceCollector, 1, SliceCollector.push, null, null, null);
    try std.testing.expect(L1.runAll(&coll, &tree));
    try std.testing.expectEqual(@as(usize, 0), coll.n);
    var ids: [4]u32 = undefined;
    var bits = IdCollector{ .buf = ids[0..] };
    try std.testing.expect(tree.iterateTargetBits(*IdCollector, &bits, IdCollector.push, IdCollector.push));
    try std.testing.expectEqual(@as(usize, 0), bits.n);
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

test "slices: L1 exact partition vs naive" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 5000, .inactive);
    tree.setRange(1000, 3000, .active);
    tree.setBit(0, true);
    tree.setBit(4999, true);
    // L1 has ceil(5000/64) = 79 ids in 2 words.
    var a_buf: [80]BitsSlice = undefined;
    var i_buf: [80]BitsSlice = undefined;
    var m_buf: [80]BitsSlice = undefined;
    var d_buf: [80]BitsSlice = undefined;
    var q = QuadCollector{ .a_buf = a_buf[0..], .i_buf = i_buf[0..], .m_buf = m_buf[0..], .d_buf = d_buf[0..] };
    const L1 = LayerSlicesIterator(*QuadCollector, 1, QuadCollector.pushA, QuadCollector.pushI, QuadCollector.pushM, QuadCollector.pushD);
    try std.testing.expect(L1.runAll(&q, &tree));
    try std.testing.expect(q.na > 0 and q.ni > 0);
    try std.testing.expect(q.nm + q.nd > 0);
    var seen = [_]bool{false} ** 79;
    try checkSliceList(a_buf[0..q.na], q.na, 1, seen[0..]);
    try checkSliceList(i_buf[0..q.ni], q.ni, 1, seen[0..]);
    try checkSliceList(m_buf[0..q.nm], q.nm, 1, seen[0..]);
    try checkSliceList(d_buf[0..q.nd], q.nd, 1, seen[0..]);
    for (seen) |v| try std.testing.expect(v);
    for (a_buf[0..q.na]) |s| try checkSlicePurity(s, false, &tree, 5000);
    for (i_buf[0..q.ni]) |s| try checkSlicePurity(s, false, &tree, 5000);
    for (m_buf[0..q.nm]) |s| try checkSlicePurity(s, true, &tree, 5000);
    for (d_buf[0..q.nd]) |s| try checkSlicePurity(s, true, &tree, 5000);
}

test "slices: chain stepSlice exact bits" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 20000, .inactive);
    tree.setRange(0, 5000, .active);
    var buf: [5000]u32 = undefined;
    var coll = IdCollector{ .buf = buf[0..] };
    const bp = BitPush{ .coll = &coll };
    const Leaf = LayerSlicesIterator(BitPush, 0, BitPush.active, null, null, null);
    const L1 = LayerSlicesIterator(BitPush, 1, BitPush.active, null, Leaf.stepSlice, null);
    const L2 = LayerSlicesIterator(BitPush, 2, BitPush.active, null, L1.stepSlice, null);
    try std.testing.expect(L2.runAll(bp, &tree));
    // Active set is exactly [0, 5000), ascending by depth-first order.
    try std.testing.expectEqual(@as(usize, 5000), coll.n);
    var k: usize = 0;
    while (k < coll.n) : (k += 1) {
        try std.testing.expectEqual(@as(u32, @intCast(k)), coll.buf[k]);
    }
    // Restricted stepSlice: L2 slice {1,2} covers bits [4096, 8192); actives are [4096, 5000).
    var coll2 = IdCollector{ .buf = buf[0..] };
    const bp2 = BitPush{ .coll = &coll2 };
    try std.testing.expect(L2.stepSlice(bp2, &tree, .{ .start = 1, .end = 2, .layer = 2 }));
    try std.testing.expectEqual(@as(usize, 904), coll2.n);
    var j: usize = 0;
    while (j < coll2.n) : (j += 1) {
        try std.testing.expectEqual(@as(u32, @intCast(4096 + j)), coll2.buf[j]);
    }
    // Out-of-range and wrong-layer slices are silent no-ops.
    var coll3 = IdCollector{ .buf = buf[0..] };
    const bp3 = BitPush{ .coll = &coll3 };
    try std.testing.expect(L2.stepSlice(bp3, &tree, .{ .start = 9999, .end = 10000, .layer = 2 }));
    try std.testing.expectEqual(@as(usize, 0), coll3.n);
    try std.testing.expect(L1.stepSlice(bp3, &tree, .{ .start = 5, .end = 6, .layer = 0 }));
    try std.testing.expectEqual(@as(usize, 0), coll3.n);
    // Leaf tail word ([19968, 20000)) is uniform inactive: nothing.
    var leaf_tail = IdCollector{ .buf = buf[0..] };
    const bpt = BitPush{ .coll = &leaf_tail };
    try std.testing.expect(Leaf.stepSlice(bpt, &tree, .{ .start = 19968, .end = 20000, .layer = 0 }));
    try std.testing.expectEqual(@as(usize, 0), leaf_tail.n);
    // Leaf word 78 ([4992, 5056)) contributes its 8 active bits.
    var leaf78 = IdCollector{ .buf = buf[0..] };
    const bp78 = BitPush{ .coll = &leaf78 };
    try std.testing.expect(Leaf.stepSlice(bp78, &tree, .{ .start = 4992, .end = 5056, .layer = 0 }));
    try std.testing.expectEqual(@as(usize, 8), leaf78.n);
    var q: usize = 0;
    while (q < leaf78.n) : (q += 1) {
        try std.testing.expectEqual(@as(u32, @intCast(4992 + q)), leaf78.buf[q]);
    }
}

test "slices: leaf both polarities exact" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 200, .inactive);
    tree.setBit(3, true);
    tree.setBit(5, true);
    tree.setBit(64, true);
    var a_buf: [8]BitsSlice = undefined;
    var i_buf: [8]BitsSlice = undefined;
    var q = QuadCollector{ .a_buf = a_buf[0..], .i_buf = i_buf[0..], .m_buf = &[_]BitsSlice{}, .d_buf = &[_]BitsSlice{} };
    const Leaf = LayerSlicesIterator(*QuadCollector, 0, QuadCollector.pushA, QuadCollector.pushI, null, null);
    try std.testing.expect(Leaf.runAll(&q, &tree));
    // Words 0..1 are fragmented, words 2..3 coalesce: 3 active + 6 inactive slices.
    try std.testing.expectEqual(@as(usize, 3), q.na);
    try std.testing.expectEqual(@as(usize, 6), q.ni);
    var seen = [_]bool{false} ** 200;
    try checkSliceList(a_buf[0..q.na], q.na, 0, seen[0..]);
    try checkSliceList(i_buf[0..q.ni], q.ni, 0, seen[0..]);
    for (seen) |v| try std.testing.expect(v);
    try std.testing.expect(tree.getBit(a_buf[0].start));
    try std.testing.expectEqual(@as(u32, 3), a_buf[0].start);
    try std.testing.expectEqual(@as(u32, 5), a_buf[1].start);
    try std.testing.expectEqual(@as(u32, 64), a_buf[2].start);
}

test "slices: four-polarity split at L2" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 20000, .inactive);
    tree.setRange(0, 5000, .active);
    var a_buf: [8]BitsSlice = undefined;
    var i_buf: [8]BitsSlice = undefined;
    var m_buf: [8]BitsSlice = undefined;
    var d_buf: [8]BitsSlice = undefined;
    var q = QuadCollector{ .a_buf = a_buf[0..], .i_buf = i_buf[0..], .m_buf = m_buf[0..], .d_buf = d_buf[0..] };
    const L2 = LayerSlicesIterator(*QuadCollector, 2, QuadCollector.pushA, QuadCollector.pushI, QuadCollector.pushM, QuadCollector.pushD);
    try std.testing.expect(L2.runAll(&q, &tree));
    // L2 has ceil(20000/4096) = 5 ids: [0]=active, [1]=shallow mixed, [2..4]=inactive.
    try std.testing.expectEqual(@as(usize, 1), q.na);
    try std.testing.expectEqual(@as(usize, 1), q.nm);
    try std.testing.expectEqual(@as(usize, 0), q.nd);
    try std.testing.expectEqual(@as(usize, 1), q.ni);
    try std.testing.expectEqual(BitsSlice{ .start = 0, .end = 1, .layer = 2 }, a_buf[0]);
    try std.testing.expectEqual(BitsSlice{ .start = 1, .end = 2, .layer = 2 }, m_buf[0]);
    try std.testing.expectEqual(BitsSlice{ .start = 2, .end = 5, .layer = 2 }, i_buf[0]);
}

test "slices: deep mixed external jump" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 8192, .inactive);
    var b: u32 = 0;
    while (b < 8192) : (b += 2) {
        tree.setBit(b, true);
    }
    try std.testing.expectEqual(@as(u32, 4096), tree.count(.active));
    // Every leaf word is fragmented, so the L2 word holds one deep run {0,2}.
    // (Same-polarity runs merge within a word; only uniform runs coalesce across words.)
    var a_buf: [8]BitsSlice = undefined;
    var i_buf: [8]BitsSlice = undefined;
    var m_buf: [8]BitsSlice = undefined;
    var d_buf: [8]BitsSlice = undefined;
    var q = QuadCollector{ .a_buf = a_buf[0..], .i_buf = i_buf[0..], .m_buf = m_buf[0..], .d_buf = d_buf[0..] };
    const L2 = LayerSlicesIterator(*QuadCollector, 2, QuadCollector.pushA, QuadCollector.pushI, QuadCollector.pushM, QuadCollector.pushD);
    try std.testing.expect(L2.runAll(&q, &tree));
    try std.testing.expectEqual(@as(usize, 0), q.na);
    try std.testing.expectEqual(@as(usize, 0), q.ni);
    try std.testing.expectEqual(@as(usize, 0), q.nm);
    try std.testing.expectEqual(@as(usize, 1), q.nd);
    try std.testing.expectEqual(BitsSlice{ .start = 0, .end = 2, .layer = 2 }, d_buf[0]);
    // External descent: the deep slice yields one deep run per L1 word, no L0 touch yet.
    var l1_buf: [128]BitsSlice = undefined;
    var l1 = SliceCollector{ .buf = l1_buf[0..] };
    const L1d = LayerSlicesIterator(*SliceCollector, 1, null, null, null, SliceCollector.push);
    try std.testing.expect(L1d.stepSlice(&l1, &tree, d_buf[0]));
    try std.testing.expectEqual(@as(usize, 2), l1.n);
    try std.testing.expectEqual(BitsSlice{ .start = 0, .end = 64, .layer = 1 }, l1.buf[0]);
    try std.testing.expectEqual(BitsSlice{ .start = 64, .end = 128, .layer = 1 }, l1.buf[1]);
    // Full external chain down to bits recovers exactly the even ids.
    var buf: [4096]u32 = undefined;
    var coll = IdCollector{ .buf = buf[0..] };
    const bp = BitPush{ .coll = &coll };
    const Leaf = LayerSlicesIterator(BitPush, 0, BitPush.active, null, null, null);
    const L1 = LayerSlicesIterator(BitPush, 1, BitPush.active, null, Leaf.stepSlice, Leaf.stepSlice);
    const L2c = LayerSlicesIterator(BitPush, 2, BitPush.active, null, L1.stepSlice, L1.stepSlice);
    try std.testing.expect(L2c.runAll(bp, &tree));
    try std.testing.expectEqual(@as(usize, 4096), coll.n);
    var k: usize = 0;
    while (k < coll.n) : (k += 1) {
        try std.testing.expectEqual(@as(u32, @intCast(2 * k)), coll.buf[k]);
    }
}

test "slices: early break stops walk" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 5000, .active);
    // Dense L1 coalesces to exactly 2 slices: {0,64} and {64,79}.
    var b1 = SliceBreakCtx{ .limit = 1 };
    const L1 = LayerSlicesIterator(*SliceBreakCtx, 1, SliceBreakCtx.cb, null, null, null);
    try std.testing.expect(!L1.runAll(&b1, &tree));
    try std.testing.expectEqual(@as(usize, 1), b1.calls);
    var b2 = SliceBreakCtx{ .limit = std.math.maxInt(usize) };
    try std.testing.expect(L1.runAll(&b2, &tree));
    try std.testing.expectEqual(@as(usize, 2), b2.calls);
}

test "slices: guards empty, shallow, out-of-range" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    var buf: [8]BitsSlice = undefined;
    var coll = SliceCollector{ .buf = buf[0..] };
    const L1 = LayerSlicesIterator(*SliceCollector, 1, SliceCollector.push, SliceCollector.push, SliceCollector.push, SliceCollector.push);
    try std.testing.expect(L1.runAll(&coll, &tree));
    try std.testing.expectEqual(@as(usize, 0), coll.n);
    try tree.resize(alloc, 100, .inactive);
    // scan_layer 5 does not exist in a depth-2 tree: silent no-op.
    var coll2 = SliceCollector{ .buf = buf[0..] };
    const Deep = LayerSlicesIterator(*SliceCollector, 5, SliceCollector.push, null, null, null);
    try std.testing.expect(Deep.runAll(&coll2, &tree));
    try std.testing.expectEqual(@as(usize, 0), coll2.n);
    // Out-of-range, wrong-layer and empty slices are silent no-ops (L1 has 2 ids here).
    var coll3 = SliceCollector{ .buf = buf[0..] };
    try std.testing.expect(L1.stepSlice(&coll3, &tree, .{ .start = 99, .end = 100, .layer = 1 }));
    try std.testing.expectEqual(@as(usize, 0), coll3.n);
    try std.testing.expect(L1.stepSlice(&coll3, &tree, .{ .start = 0, .end = 1, .layer = 0 }));
    try std.testing.expectEqual(@as(usize, 0), coll3.n);
    try std.testing.expect(L1.stepSlice(&coll3, &tree, .{ .start = 1, .end = 1, .layer = 1 }));
    try std.testing.expectEqual(@as(usize, 0), coll3.n);
    const Leaf = LayerSlicesIterator(*SliceCollector, 0, SliceCollector.push, null, null, null);
    var coll4 = SliceCollector{ .buf = buf[0..] };
    try std.testing.expect(Leaf.stepSlice(&coll4, &tree, .{ .start = 1000, .end = 1001, .layer = 0 }));
    try std.testing.expectEqual(@as(usize, 0), coll4.n);
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
    var bit: u32 = 0;
    while (bit < 5000) : (bit += 1) {
        const want = (bit >= 1000 and bit < 4000) or bit == 0 or bit == 4999;
        if (!want) continue;
        try std.testing.expect(k < coll.n);
        try std.testing.expectEqual(bit, coll.buf[k]);
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

test "sumActiveIds matches naive sum" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 20000, .inactive);
    tree.setRange(0, 5000, .active);
    tree.setBit(19999, true);
    const got = tree.sumActiveIds();
    var want_sum: u64 = 0;
    var want_count: u64 = 0;
    var bit: u32 = 0;
    while (bit < 20000) : (bit += 1) {
        if (tree.getBit(bit)) {
            want_sum += bit;
            want_count += 1;
        }
    }
    try std.testing.expectEqual(want_count, got.count);
    try std.testing.expectEqual(want_sum, got.sum);
    try std.testing.expectEqual(@as(u64, 5001), got.count);
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
            var s_k: usize = 0;
            while (s_k < n) : (s_k += 1) {
                try std.testing.expect(buf[s_k] < model_len);
                try std.testing.expect(model[buf[s_k]]);
                try std.testing.expect(!seen[buf[s_k]]);
                seen[buf[s_k]] = true;
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
