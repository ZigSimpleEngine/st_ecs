/// Bitset with a hierarchy of packed u2 regions for fast skipping of uniform spans.
/// Level 0 holds bits, 64 per u64. Higher levels hold summaries, 32 regions per u64.
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

/// State of one region in the hierarchy.
/// Densely packed: 2 bits per region, 32 regions per summary u64 word.
/// The reserved value 3 is treated as mixed when read.
const NodeState = enum(u2) {
    /// Every bit of the region is cleared.
    inactive = 0,
    /// Every bit of the region is set.
    active = 1,
    /// Region holds both cleared and set bits.
    mixed = 2,
};

/// One hierarchy level stored as packed 64-bit words.
const Level = ListA64(u64);

/// Bits per leaf word (level 0). Exactly one u64 register, so ctz and popcount run in a tick.
pub const leaf_fanout: u32 = 64;
/// Regions per summary word (levels 1 and above): 32 x u2 inside one u64.
/// 32 keeps a node in a single register with depth log32(N/64).
/// A fanout of 16 doubles the depth, a fanout of 64 needs 128 bits per node.
pub const node_fanout: u32 = 32;

/// Maximum level count for u32 indices: 2^32 bits need 2^26 leaf words,
/// then 2^21, 2^16, 2^11, 2^6, 2 and 1 words, which is 7 levels. 8 leaves headroom.
const max_levels: usize = 8;

/// Word with every bit set, used as fill and mask value.
const all_ones: u64 = std.math.maxInt(u64);

/// Bit plane selecting the low bit of every 2-bit summary slot.
/// Used to collapse 32 packed slot states into 32 side-by-side match bits.
const slot_lo_plane: u64 = 0x5555_5555_5555_5555;

/// Bit span of one region summarized at each summary level.
/// A slot at summary level L covers level_span[L - 1] bits.
const level_span: [max_levels]u64 = blk: {
    var table: [max_levels]u64 = undefined;
    var span: u64 = leaf_fanout;
    for (&table) |*entry| {
        entry.* = span;
        span *= node_fanout;
    }
    break :blk table;
};

/// Counts leaf words needed to hold the given bit count.
/// - `bits` total bit count.
///
/// Return: number of u64 leaf words.
inline fn leafWordsFor(bits: u32) usize {
    return std.math.divCeil(u32, bits, leaf_fanout) catch unreachable;
}

/// Counts parent summary words needed to cover the given child word count.
/// - `child_words` word count on the child level.
///
/// Return: number of u64 summary words.
inline fn parentWordsFor(child_words: usize) usize {
    return std.math.divCeil(usize, child_words, node_fanout) catch unreachable;
}

/// Counts hierarchy levels needed for the given bit count.
/// - `bits` total bit count.
///
/// Return: level count, 0 when there are no bits.
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

/// Counts words on one level for the given bit count.
/// - `level` hierarchy level, 0 is the leaf bitset.
/// - `bits` total bit count.
///
/// Return: word count on the requested level.
fn wordsForLevel(level: usize, bits: u32) usize {
    var words: usize = leafWordsFor(bits);
    var l: usize = 0;
    while (l < level) : (l += 1) {
        words = parentWordsFor(words);
    }
    return words;
}

/// Builds the valid-bit mask for the last leaf word.
/// Tail bits past the total count are always zero.
/// - `total_bits` total bit count.
///
/// Return: mask with one bit per valid tail bit.
inline fn lastLeafMask(total_bits: u32) u64 {
    const r: u6 = @intCast(total_bits & 63);
    if (r == 0) return all_ones;
    return (@as(u64, 1) << r) - 1;
}

/// Builds the mask of word bits covered by a half-open bit range.
/// - `word_idx` leaf word index.
/// - `range_start` first covered bit.
/// - `range_end` one past the last covered bit.
///
/// Return: mask with one bit per covered word bit.
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

/// Reads one packed 2-bit region state from a summary word.
/// The reserved value 3 is reported as mixed.
/// - `word` summary word holding 32 packed states.
/// - `slot` region index inside the word, 0 to 31.
///
/// Return: decoded region state.
inline fn getSlot(word: u64, slot: u32) NodeState {
    const v: u2 = @intCast((word >> @as(u6, @intCast(slot * 2))) & 0b11);
    return switch (v) {
        0 => .inactive,
        1 => .active,
        else => .mixed,
    };
}

/// Writes one packed 2-bit region state into a summary word.
/// - `word` summary word to update.
/// - `slot` region index inside the word, 0 to 31.
/// - `state` region state to store.
inline fn setSlot(word: *u64, slot: u32, state: NodeState) void {
    const shift: u6 = @intCast(slot * 2);
    word.* = (word.* & ~(@as(u64, 0b11) << shift)) | (@as(u64, @intFromEnum(state)) << shift);
}

/// Summarizes one leaf word into a region state.
/// - `word` raw leaf word.
/// - `valid_mask` mask of bits that belong to the bitset.
///
/// Return: inactive when no valid bit is set, active when all are set, mixed otherwise.
inline fn summarizeLeaf(word: u64, valid_mask: u64) NodeState {
    const w = word & valid_mask;
    if (w == 0) return .inactive;
    if (w == valid_mask) return .active;
    return .mixed;
}

/// Summarizes one summary word from its valid slots.
/// - `word` summary word holding packed states.
/// - `valid_slots` number of slots that cover real children, 1 to 32.
///
/// Return: inactive when all valid slots are inactive, active when all are active, mixed otherwise.
fn summarizeNode(word: u64, valid_slots: u32) NodeState {
    std.debug.assert(valid_slots >= 1 and valid_slots <= node_fanout);
    var seen_active = false;
    var seen_inactive = false;
    var i: u32 = 0;
    while (i < valid_slots) : (i += 1) {
        switch (getSlot(word, i)) {
            .active => seen_active = true,
            .inactive => seen_inactive = true,
            .mixed => return .mixed,
        }
        if (seen_active and seen_inactive) return .mixed;
    }
    if (seen_active) return .active;
    return .inactive;
}

/// Collapses 32 packed 2-bit slot states into 32 side-by-side match bits.
/// Bit 2*i of the result is set when slot i equals the wanted state or is mixed.
/// Branchless SWAR evaluation over the low and high bit planes of every slot.
/// - `word` summary word holding 32 packed states.
/// - `want` wanted region state.
///
/// Return: one match bit per slot stored at that slot's even bit position.
inline fn matchMask(word: u64, want: NodeState) u64 {
    return switch (want) {
        .active => (word | (word >> 1)) & slot_lo_plane,
        .inactive => (~word | (word >> 1)) & slot_lo_plane,
        .mixed => (word >> 1) & slot_lo_plane,
    };
}

/// Finds the first slot of a summary word, scanned from `from` up to `limit`,
/// whose state equals the wanted state or is mixed. Uniform regions are skipped.
/// Evaluated branchless: one SWAR match over the whole word plus a single ctz.
/// - `word` summary word holding packed states.
/// - `from` first scanned slot index. Values at or above `limit` match nothing.
/// - `limit` one past the last scanned slot index.
/// - `want` wanted region state.
///
/// Return: matching slot index, or null when no scanned slot matches.
inline fn scanSlots(word: u64, from: u32, limit: u32, want: NodeState) ?u32 {
    std.debug.assert(limit <= node_fanout);
    if (from >= limit) return null;
    var m: u64 = matchMask(word, want);
    if (from != 0) {
        const lo: u6 = @intCast(from * 2);
        m &= all_ones << lo;
    }
    if (limit != node_fanout) {
        const hi: u6 = @intCast(limit * 2);
        m &= (@as(u64, 1) << hi) - 1;
    }
    if (m == 0) return null;
    return @as(u32, @ctz(m)) >> 1;
}

/// Bitset with a hierarchy of packed u2 regions for fast skipping of uniform spans.
/// Level 0 holds bits, 64 per u64. Higher levels hold summaries, 32 regions per u64.
pub const BitTree = struct {
    /// Short alias of the enclosing type.
    const Self = @This();

    /// Number of addressable bits in the bitset.
    total_bits: u32 = 0,
    /// Number of set bits. Updated by every mutating operation.
    active_count: u32 = 0,
    /// Hierarchy levels. Index 0 is the leaf bitset, each higher level summarizes the one below.
    levels: ListA64(Level) = .empty,

    /// Empty tree with no bits and no allocated levels.
    pub const empty: Self = .{};

    /// Releases every level buffer and invalidates the tree.
    /// - `self` tree to destroy.
    /// - `allocator` allocator that owns the level buffers.
    pub fn deinit(self: *Self, allocator: Allocator) void {
        for (self.levels.items) |*lvl| {
            lvl.deinit(allocator);
        }
        self.levels.deinit(allocator);
        self.* = undefined;
    }

    /// Reports the number of addressable bits.
    /// - `self` inspected tree.
    ///
    /// Return: total bit count.
    pub inline fn totalBitsCount(self: *const Self) u32 {
        return self.total_bits;
    }

    /// Counts bits in the given state in O(1) through the maintained counter.
    /// - `self` inspected tree.
    /// - `state` bit state to count.
    ///
    /// Return: number of bits in the requested state.
    pub inline fn count(self: *const Self, state: BitState) u32 {
        return switch (state) {
            .active => self.active_count,
            .inactive => self.total_bits - self.active_count,
        };
    }

    /// Computes the sparseness factor of bits in the wanted state.
    /// Defined as the median gap between consecutive set bits, measured in bits.
    /// Returns zero when fewer than two bits are set. Manual call by design:
    /// the value is not maintained incrementally, recompute it after bulk updates.
    /// - `self` inspected tree.
    /// - `allocator` allocator for the temporary gap buffer, freed before return.
    /// - `want` bit state to measure, compile-time known.
    ///
    /// Return: median consecutive-bit gap in bits, or zero when undefined.
    pub fn sparseFactor(self: *const Self, allocator: Allocator, comptime want: BitState) Allocator.Error!u32 {
        const found: u32 = self.count(want);
        if (found < 2) return 0;
        const gaps: []u32 = try allocator.alloc(u32, @as(usize, @intCast(found - 1)));
        defer allocator.free(gaps);
        var it = self.cursor(want);
        var prev: u32 = it.step() orelse return 0;
        var n: usize = 0;
        while (it.step()) |b| {
            gaps[n] = b - prev;
            prev = b;
            n += 1;
        }
        std.debug.assert(n == gaps.len);
        std.mem.sortUnstable(u32, gaps, {}, std.sort.asc(u32));
        return gaps[(gaps.len - 1) / 2];
    }

    /// Modeled cost of one hierarchy descent serving a segment, in nanoseconds.
    const detect_cost_descent: f64 = 20.0;
    /// Modeled cost of emitting one bit from a uniform run, in nanoseconds.
    const detect_cost_run_bit: f64 = 0.6;
    /// Modeled cost of consuming one matching bit outside runs, in nanoseconds.
    const detect_cost_take_bit: f64 = 1.2;
    /// Modeled cost of checking one leaf word linearly, in nanoseconds.
    const detect_cost_word: f64 = 0.4;
    /// Level-1 words sampled at most by automatic scan selection.
    const detect_sample_words: usize = 32;

    /// Per-slot match masks of one summary word, one bit per slot at even positions.
    const SlotMasks = struct {
        /// Slots exactly equal to the wanted state.
        match: u64,
        /// Slots in the mixed state, reserved value included.
        mixed: u64,
    };

    /// Classifies valid slots of one summary word with branchless SWAR masks.
    /// Invalid slots past the valid count are always cleared first.
    /// - `word` summary word holding packed states.
    /// - `valid` number of valid slots, 1 to 32.
    /// - `want` wanted region state, compile-time known.
    ///
    /// Return: exact-want and mixed match masks.
    inline fn classifyWord(word: u64, valid: u32, comptime want: NodeState) SlotMasks {
        var vm: u64 = all_ones;
        if (valid != node_fanout) {
            vm = (@as(u64, 1) << @as(u6, @intCast(valid * 2))) - 1;
        }
        const mixed: u64 = ((word >> 1) & slot_lo_plane) & vm;
        const lo: u64 = word & slot_lo_plane;
        const hi: u64 = (word >> 1) & slot_lo_plane;
        const match: u64 = switch (want) {
            .active => (lo & ~hi) & vm,
            .inactive => (~lo & ~hi) & vm,
            .mixed => ((word >> 1) & slot_lo_plane) & vm,
        };
        return .{ .match = match, .mixed = mixed };
    }

    /// Automatically selects the faster scan strategy for bits in the wanted state.
    /// Reads the single root word: fully uniform data decides immediately.
    /// Otherwise strided-samples level 1 directly, the terminal level whose slots
    /// are leaves, and extrapolates the cost model. No frontier, no pushes,
    /// no allocation: one root read plus a fixed sample, well under a microsecond
    /// regardless of the total size. Sampling noise flips only near-tie decisions
    /// whose stakes are bounded near 1.3x either way.
    /// No tree mutation, safe to call before any query.
    /// - `self` inspected tree.
    /// - `want` bit state to scan, compile-time known.
    ///
    /// Return: recommended scan strategy.
    pub fn detectScanKind(self: *const Self, comptime want: BitState) ScanKind {
        const total = self.total_bits;
        if (total == 0) return .linear;
        const nlevels = self.levels.items.len;
        if (nlevels <= 1) return .linear;
        const want_node: NodeState = if (want == .active) .active else .inactive;
        const active: u64 = self.count(want);
        const total_words: u64 = self.levels.items[0].items.len;
        const active_f: f64 = @as(f64, @floatFromInt(active));

        const top: usize = nlevels - 1;
        const top_len: usize = self.levels.items[top].items.len;
        const below_len: usize = self.levels.items[top - 1].items.len;
        var t: usize = 0;
        while (t < top_len) : (t += 1) {
            const word: u64 = self.levels.items[top].items[t];
            const valid: u32 = @intCast(@min(below_len - t * node_fanout, @as(usize, node_fanout)));
            const masks = classifyWord(word, valid, want_node);
            if (masks.mixed != 0) break;
        }
        if (t >= top_len) {
            return if (active > 0) .linear else .tree;
        }

        const l1len: usize = self.levels.items[1].items.len;
        const leaf_len: usize = self.levels.items[0].items.len;
        var run_bits: f64 = 0.0;
        var segments: f64 = 0.0;
        var resolved_mixed: f64 = 0.0;
        var frame_len: usize = l1len;
        if (frame_len > 0) {
            frame_len -= 1;
            const w: usize = frame_len;
            const word: u64 = self.levels.items[1].items[w];
            const first_child: usize = w * node_fanout;
            const valid: u32 = @intCast(@min(leaf_len - first_child, @as(usize, node_fanout)));
            var s: u32 = 0;
            while (s < valid) : (s += 1) {
                const st: NodeState = getSlot(word, s);
                if (st == want_node) {
                    const base: u64 = @as(u64, first_child + s) * leaf_fanout;
                    run_bits += @as(f64, @floatFromInt(@min(base + leaf_fanout, @as(u64, total)) - base));
                    segments += 1.0;
                } else if (st == .mixed) {
                    resolved_mixed += 1.0;
                }
            }
        }
        if (frame_len > 0) {
            const take: usize = @min(detect_sample_words, frame_len);
            const scale: f64 = @as(f64, @floatFromInt(frame_len)) / @as(f64, @floatFromInt(take));
            var j: usize = 0;
            while (j < take) : (j += 1) {
                const w: usize = (j * frame_len) / take;
                const word: u64 = self.levels.items[1].items[w];
                const first_child: usize = w * node_fanout;
                const valid: u32 = @intCast(@min(leaf_len - first_child, @as(usize, node_fanout)));
                const masks = classifyWord(word, valid, want_node);
                run_bits += @as(f64, @floatFromInt(@popCount(masks.match))) * @as(f64, @floatFromInt(leaf_fanout)) * scale;
                segments += @as(f64, @floatFromInt(@popCount(masks.match))) * scale;
                resolved_mixed += @as(f64, @floatFromInt(@popCount(masks.mixed))) * scale;
            }
        }

        const run_capped: f64 = @min(run_bits, active_f);
        const tree_est: f64 = (segments + resolved_mixed) * detect_cost_descent +
            run_capped * detect_cost_run_bit +
            (active_f - run_capped) * detect_cost_take_bit;
        const linear_est: f64 = @as(f64, @floatFromInt(total_words)) * detect_cost_word +
            active_f * detect_cost_take_bit;
        return if (tree_est < linear_est) .tree else .linear;
    }

    /// Reads a single bit.
    /// - `self` inspected tree.
    /// - `bit` bit index, must be below the total bit count.
    ///
    /// Return: state of the requested bit.
    pub fn get(self: *const Self, bit: u32) BitState {
        std.debug.assert(bit < self.total_bits);
        const w: u64 = self.levels.items[0].items[bit >> 6];
        const s: u6 = @intCast(bit & 63);
        return if ((w >> s) & 1 == 1) .active else .inactive;
    }

    /// Sets a single bit and refreshes the whole ancestor chain.
    /// Stops early once a parent slot stays unchanged, because ancestors above cannot change either.
    /// - `self` mutated tree.
    /// - `bit` bit index, must be below the total bit count.
    /// - `state` state to store.
    pub fn set(self: *Self, bit: u32, state: BitState) void {
        std.debug.assert(bit < self.total_bits);
        std.debug.assert(self.levels.items.len > 0);
        const word_idx: usize = bit >> 6;
        const shift: u6 = @intCast(bit & 63);
        const mask: u64 = @as(u64, 1) << shift;
        const leaf: *u64 = &self.levels.items[0].items[word_idx];
        const was_active = (leaf.* & mask) != 0;
        const want_active = state == .active;
        if (was_active == want_active) return;
        if (want_active) {
            leaf.* |= mask;
            self.active_count += 1;
        } else {
            leaf.* &= ~mask;
            self.active_count -= 1;
        }
        self.propagateFromLeafWord(word_idx);
    }

    /// Refreshes the ancestor chain of one leaf word after its content changed.
    /// - `self` mutated tree.
    /// - `leaf_word_idx` changed leaf word index.
    fn propagateFromLeafWord(self: *Self, leaf_word_idx: usize) void {
        const leaf_words = self.levels.items[0].items;
        var child_idx: usize = leaf_word_idx;
        var child_state: NodeState = summarizeLeaf(
            leaf_words[child_idx],
            if (child_idx + 1 == leaf_words.len) lastLeafMask(self.total_bits) else all_ones,
        );
        var lvl: usize = 1;
        while (lvl < self.levels.items.len) : (lvl += 1) {
            const child_count: usize = self.levels.items[lvl - 1].items.len;
            const parent_items = self.levels.items[lvl].items;
            const p_idx: usize = child_idx / node_fanout;
            const slot: u32 = @intCast(child_idx % node_fanout);
            if (getSlot(parent_items[p_idx], slot) == child_state) return;
            setSlot(&parent_items[p_idx], slot, child_state);
            const remain: usize = child_count - p_idx * node_fanout;
            const valid: u32 = @intCast(@min(remain, @as(usize, node_fanout)));
            child_state = summarizeNode(parent_items[p_idx], valid);
            child_idx = p_idx;
        }
    }

    /// Sets a bit range to one state. Touches covered leaf words only,
    /// then propagates upward per word for small ranges or rebuilds every summary for large ones.
    /// - `self` mutated tree.
    /// - `start` first bit index of the range.
    /// - `len` number of covered bits, may be zero for a no-op.
    /// - `state` state to store.
    pub fn setRange(self: *Self, start: u32, len: u32, state: BitState) void {
        if (len == 0) return;
        std.debug.assert(self.levels.items.len > 0);
        std.debug.assert(@as(u64, start) + len <= self.total_bits);
        const end: u64 = @as(u64, start) + len;
        const first_word: usize = @intCast(start >> 6);
        const last_word: usize = @intCast((end - 1) >> 6);
        const leaves = self.levels.items[0].items;
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
    /// - `self` mutated tree.
    /// - `state` state to store into every bit.
    pub fn clear(self: *Self, state: BitState) void {
        if (self.total_bits == 0) return;
        const leaves = self.levels.items[0].items;
        @memset(leaves, if (state == .active) all_ones else 0);
        if (self.total_bits & 63 != 0) {
            leaves[leaves.len - 1] &= lastLeafMask(self.total_bits);
        }
        self.active_count = if (state == .active) self.total_bits else 0;
        self.rebuildSummaries();
    }

    /// Precise resize. Excess memory is really returned to the allocator,
    /// through precise allocation on growth and shrinking on contraction.
    /// New bits are initialized with the requested state.
    /// - `self` mutated tree.
    /// - `allocator` allocator that owns the level buffers.
    /// - `new_bits_count` requested total bit count.
    /// - `new_bits_state` initial state of bits added by growth.
    pub fn resize(self: *Self, allocator: Allocator, new_bits_count: u32, new_bits_state: BitState) !void {
        try self.resizeImpl(allocator, new_bits_count, new_bits_state, .precise);
    }

    /// Resize that retains capacity. Contraction keeps allocated memory
    /// and growth follows the geometric list policy, so fluctuating sizes avoid reallocations.
    /// Fully removed upper levels are always freed. New bits are initialized with the requested state.
    /// - `self` mutated tree.
    /// - `allocator` allocator that owns the level buffers.
    /// - `new_bits_count` requested total bit count.
    /// - `new_bits_state` initial state of bits added by growth.
    pub fn resizeRetainingCapacity(self: *Self, allocator: Allocator, new_bits_count: u32, new_bits_state: BitState) !void {
        try self.resizeImpl(allocator, new_bits_count, new_bits_state, .retaining);
    }

    /// Memory policy shared by both public resize entry points.
    const ResizeMode = enum {
        /// Free excess memory and allocate grown buffers exactly.
        precise,
        /// Keep excess capacity and grow geometrically.
        retaining,
    };

    /// Shared resize implementation behind the two public entry points.
    /// Reserves capacity first without touching lengths, so an allocation failure leaves the tree intact,
    /// then applies lengths, fills leaf content, trims removed levels and rebuilds every summary.
    /// - `self` mutated tree.
    /// - `allocator` allocator that owns the level buffers.
    /// - `new_bits_count` requested total bit count.
    /// - `new_bits_state` initial state of bits added by growth.
    /// - `mode` memory policy selected at compile time.
    fn resizeImpl(self: *Self, allocator: Allocator, new_bits_count: u32, new_bits_state: BitState, comptime mode: ResizeMode) !void {
        const old_bits = self.total_bits;
        if (new_bits_count == old_bits) {
            if (mode == .precise) {
                for (self.levels.items) |*lvl| {
                    if (lvl.capacity != lvl.items.len) lvl.shrinkAndFree(allocator, lvl.items.len);
                }
                if (self.levels.capacity != self.levels.items.len) {
                    self.levels.shrinkAndFree(allocator, self.levels.items.len);
                }
            }
            return;
        }

        const old_depth: usize = self.levels.items.len;
        const old_leaf_words: usize = if (old_depth > 0) self.levels.items[0].items.len else 0;

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
                    .precise => try self.levels.items[l].ensureTotalCapacityPrecise(allocator, sizes[l]),
                    .retaining => try self.levels.items[l].ensureTotalCapacity(allocator, sizes[l]),
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
            new_levels[new_count] = try Level.initCapacity(allocator, sizes[l]);
            new_count += 1;
        }

        l = 0;
        while (l < old_depth and l < need_depth) : (l += 1) {
            try self.levels.items[l].resize(allocator, sizes[l]);
        }
        l = 0;
        while (l < new_count) : (l += 1) {
            try new_levels[l].resize(allocator, sizes[old_depth + l]);
            @memset(new_levels[l].items, 0);
            self.levels.appendAssumeCapacity(new_levels[l]);
        }
        new_count = 0;

        self.total_bits = new_bits_count;

        var added_active: u32 = 0;
        if (new_bits_count > old_bits) {
            const leaves = self.levels.items[0].items;
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
                const leaves = self.levels.items[0].items;
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
                if (lvl.capacity != lvl.items.len) lvl.shrinkAndFree(allocator, lvl.items.len);
            }
        }
    }

    /// Counts set bits inside a bit range.
    /// - `self` inspected tree.
    /// - `start` first bit index of the range.
    /// - `len` number of covered bits.
    ///
    /// Return: number of set bits inside the range.
    fn countRangeActive(self: *const Self, start: u32, len: u32) u32 {
        if (len == 0) return 0;
        const leaves = self.levels.items[0].items;
        const end: u64 = @as(u64, start) + len;
        var w: usize = @intCast(start >> 6);
        const last_word: usize = @intCast((end - 1) >> 6);
        var acc: u32 = 0;
        while (w <= last_word) : (w += 1) {
            acc += @intCast(@popCount(leaves[w] & rangeMask(w, start, end)));
        }
        return acc;
    }

    /// Recomputes every summary bottom-up. Cold path used by resize, clear and large range writes.
    /// - `self` mutated tree.
    fn rebuildSummaries(self: *Self) void {
        const nlevels = self.levels.items.len;
        var lvl: usize = 1;
        while (lvl < nlevels) : (lvl += 1) {
            const child_len: usize = self.levels.items[lvl - 1].items.len;
            const parent_items = self.levels.items[lvl].items;
            var p: usize = 0;
            while (p < parent_items.len) : (p += 1) {
                var word: u64 = 0;
                const first_child: usize = p * node_fanout;
                const valid: u32 = @intCast(@min(child_len - first_child, @as(usize, node_fanout)));
                var s: u32 = 0;
                while (s < valid) : (s += 1) {
                    const ci: usize = first_child + s;
                    const cs: NodeState = if (lvl == 1) blk: {
                        const leaves = self.levels.items[0].items;
                        const mask: u64 = if (ci + 1 == leaves.len) lastLeafMask(self.total_bits) else all_ones;
                        break :blk summarizeLeaf(leaves[ci], mask);
                    } else blk: {
                        const cw: u64 = self.levels.items[lvl - 1].items[ci];
                        const below_len: usize = self.levels.items[lvl - 2].items.len;
                        const cvalid: u32 = @intCast(@min(below_len - ci * node_fanout, @as(usize, node_fanout)));
                        break :blk summarizeNode(cw, cvalid);
                    };
                    setSlot(&word, s, cs);
                }
                parent_items[p] = word;
            }
        }
    }

    /// Finds the first bit in the wanted state strictly after the start bit.
    /// A null start searches from bit zero inclusively.
    /// Uniform regions are skipped through the summaries. Missing match returns null.
    /// Implemented as a single step of a cursor positioned past the start bit.
    /// - `self` inspected tree.
    /// - `start_bit` bit index to search after, or null to start from the beginning.
    /// - `want` wanted bit state.
    ///
    /// Return: index of the first matching bit, or null when there is none.
    pub fn next(self: *const Self, start_bit: ?u32, want: BitState) ?u32 {
        const total = self.total_bits;
        if (total == 0) return null;
        var pos: u64 = 0;
        if (start_bit) |s| {
            const p: u64 = @as(u64, s) + 1;
            if (p >= total) return null;
            pos = p;
        }
        var it = SkipCursor{ .tree = self, .want = want, .pos = pos };
        return it.step();
    }

    /// Single descent outcome: either a uniform matching region or a mixed leaf word.
    const FindResult = union(enum) {
        /// Uniform region fully matching the query. Emit by counter from start to end.
        run: struct { start: u64, end: u64 },
        /// Mixed chain bottomed out here. Scan the word bits.
        leaf: usize,
    };

    /// Locates the first match at or after a bit position.
    /// Descends top-down, scanning summary slots forward from the entry point of the position.
    /// A slot uniform in the wanted state returns its whole region for counter emission,
    /// without touching any leaf below. Only mixed slots descend further.
    /// A fully scanned word backtracks to its parent and resumes after the slot descended through,
    /// so every visited word is vetted by its parent and the worst case stays logarithmic.
    /// Only the top level advances word by word, bounded by 32 words.
    /// - `self` inspected tree.
    /// - `from_pos` bit position where the search starts.
    /// - `want` wanted region state.
    ///
    /// Return: uniform run or candidate leaf word, or null when no region below can match.
    fn findNext(self: *const Self, from_pos: u64, want: NodeState) ?FindResult {
        const nlevels = self.levels.items.len;
        if (nlevels == 0) return null;
        const first_word: usize = @intCast(from_pos / leaf_fanout);
        if (nlevels == 1) {
            if (first_word >= self.levels.items[0].items.len) return null;
            const only: u64 = self.levels.items[0].items[first_word];
            const mask: u64 = if (first_word + 1 == self.levels.items[0].items.len) lastLeafMask(self.total_bits) else all_ones;
            if (summarizeLeaf(only, mask) == want) {
                const base: u64 = @as(u64, first_word) * leaf_fanout;
                const start: u64 = @max(base, from_pos);
                const end: u64 = @min(base + level_span[0], @as(u64, self.total_bits));
                if (start < end) return .{ .run = .{ .start = start, .end = end } };
            }
            return .{ .leaf = first_word };
        }
        std.debug.assert(nlevels <= max_levels);
        var entry: [max_levels]usize = undefined;
        entry[0] = first_word;
        var l: usize = 1;
        while (l < nlevels) : (l += 1) {
            entry[l] = entry[l - 1] / node_fanout;
        }
        var lvl: usize = nlevels - 1;
        var word_idx: usize = entry[lvl];
        var slot_from: u32 = @intCast(entry[lvl - 1] % node_fanout);
        while (true) {
            const words = self.levels.items[lvl].items;
            if (word_idx < words.len) {
                const child_len: usize = self.levels.items[lvl - 1].items.len;
                const limit: u32 = @intCast(@min(child_len - word_idx * node_fanout, @as(usize, node_fanout)));
                if (scanSlots(words[word_idx], slot_from, limit, want)) |slot| {
                    const child: usize = word_idx * node_fanout + slot;
                    if (getSlot(words[word_idx], slot) == want) {
                        const span: u64 = level_span[lvl - 1];
                        const base: u64 = @as(u64, child) * span;
                        const start: u64 = @max(base, from_pos);
                        const end: u64 = @min(base + span, @as(u64, self.total_bits));
                        if (start < end) return .{ .run = .{ .start = start, .end = end } };
                        slot_from = slot + 1;
                        continue;
                    }
                    if (lvl == 1) return .{ .leaf = child };
                    lvl -= 1;
                    word_idx = child;
                    slot_from = if (word_idx == entry[lvl]) @intCast(entry[lvl - 1] % node_fanout) else 0;
                    continue;
                }
                if (lvl + 1 >= nlevels) {
                    word_idx += 1;
                    slot_from = 0;
                    continue;
                }
                slot_from = @as(u32, @intCast(word_idx % node_fanout)) + 1;
                word_idx = word_idx / node_fanout;
                lvl += 1;
                continue;
            }
            return null;
        }
    }

    /// Builds the masked content of one leaf word for a scan in the wanted state.
    /// Bits before the position and tail bits past the total count are cleared.
    /// - `self` inspected tree.
    /// - `leaf_word_idx` scanned leaf word index.
    /// - `from_pos` bit position where the scan starts.
    /// - `want` wanted bit state.
    ///
    /// Return: leaf word with only not-yet-visited candidate bits set.
    fn maskedLeafWord(self: *const Self, leaf_word_idx: usize, from_pos: u64, want: BitState) u64 {
        const leaves = self.levels.items[0].items;
        if (leaf_word_idx >= leaves.len) return 0;
        var w: u64 = leaves[leaf_word_idx];
        if (want == .inactive) w = ~w;
        const base: u64 = @as(u64, leaf_word_idx) * leaf_fanout;
        if (from_pos > base) {
            const skip: u6 = @intCast(from_pos - base);
            if (skip != 0) w &= all_ones << skip;
        }
        if (leaf_word_idx + 1 == leaves.len and (self.total_bits & 63) != 0) {
            w &= lastLeafMask(self.total_bits);
        }
        return w;
    }

    /// Stateful forward cursor over bits in one state. Primary query path for full scans.
    /// Caches the masked content of the current leaf word, so consecutive bits inside one
    /// word cost a single ctz plus a bit clear each, without any hierarchy descent.
    /// The hierarchy is consulted only when the cached word is exhausted.
    pub const SkipCursor = struct {
        /// Inspected tree. Must outlive the cursor and must not be mutated during iteration.
        tree: *const BitTree,
        /// Bit state produced by the cursor.
        want: BitState,
        /// Bit position where the next step starts examining. Inclusive.
        pos: u64 = 0,
        /// Not-yet-returned candidate bits of the cached word. Meaningful only with `has_cache`.
        bits: u64 = 0,
        /// Leaf word index backing `bits`.
        cached_word: usize = 0,
        /// Whether `bits` and `cached_word` hold a usable cached word.
        has_cache: bool = false,
        /// One past the end of the uniform matching run under emission.
        /// A run is active exactly while `pos` is below `run_end`.
        run_end: u64 = 0,

        /// Moves the examination position. Invalidates the cached word and any active run.
        /// Monotonic forward stepping without seek calls is the fast path.
        /// - `self` mutated cursor.
        /// - `pos` bit position where examination resumes. Inclusive.
        pub fn seek(self: *SkipCursor, pos: u64) void {
            self.pos = pos;
            self.has_cache = false;
            self.run_end = pos;
        }

        /// Returns the next matching bit at or after the examination position.
        /// Advances past the returned bit, clearing it from the cached word, so the
        /// following call resumes inside the same word whenever bits are left.
        /// Bits inside the cached word are served without any hierarchy descent,
        /// and uniform regions are emitted by counter without reading any leaf at all.
        /// - `self` mutated cursor.
        ///
        /// Return: next matching bit index, or null when iteration is complete.
        pub inline fn step(self: *SkipCursor) ?u32 {
            const tree = self.tree;
            const total = tree.total_bits;
            if (total == 0) return null;
            const want_node: NodeState = switch (self.want) {
                .active => .active,
                .inactive => .inactive,
            };
            while (self.pos < total) {
                if (self.has_cache and (self.pos >> 6) == @as(u64, self.cached_word)) {
                    if (self.bits == 0) {
                        self.pos = (@as(u64, self.cached_word) + 1) * @as(u64, leaf_fanout);
                        self.has_cache = false;
                        continue;
                    }
                    return self.takeCachedBit(total);
                }
                if (self.pos < self.run_end) {
                    const b: u32 = @intCast(self.pos);
                    self.pos += 1;
                    return b;
                }
                self.has_cache = false;
                const found = tree.findNext(self.pos, want_node) orelse return null;
                switch (found) {
                    .run => |r| {
                        self.pos = r.start;
                        self.run_end = r.end;
                        continue;
                    },
                    .leaf => |lw| {
                        self.bits = tree.maskedLeafWord(lw, self.pos, self.want);
                        self.cached_word = lw;
                        self.has_cache = true;
                        self.run_end = self.pos;
                        if (self.bits == 0) {
                            self.pos = (@as(u64, lw) + 1) * @as(u64, leaf_fanout);
                            self.has_cache = false;
                            continue;
                        }
                        const base: u64 = @as(u64, lw) * @as(u64, leaf_fanout);
                        if (self.pos < base) self.pos = base;
                        self.run_end = self.pos;
                    },
                }
            }
            return null;
        }

        /// Returns the lowest set bit of the cached word and clears it from the cache.
        /// The caller guarantees a warm cache positioned inside the cached word.
        /// - `self` mutated cursor.
        /// - `total` total bit count bounding the result.
        ///
        /// Return: next matching bit index, or null on a defeated tail guard.
        inline fn takeCachedBit(self: *SkipCursor, total: u32) ?u32 {
            const b: u32 = @intCast(@as(u64, self.cached_word) * @as(u64, leaf_fanout) + @as(u64, @ctz(self.bits)));
            if (b >= total) return null;
            self.bits &= self.bits - 1;
            if (self.bits == 0) {
                self.pos = (@as(u64, self.cached_word) + 1) * @as(u64, leaf_fanout);
                self.has_cache = false;
            } else {
                self.pos = @as(u64, b) + 1;
            }
            return b;
        }
    };

    /// Creates a forward cursor over bits in the wanted state, positioned at bit zero.
    /// - `self` inspected tree.
    /// - `want` bit state produced by the cursor.
    ///
    /// Return: cursor positioned before the first matching bit.
    pub fn cursor(self: *const Self, want: BitState) SkipCursor {
        return .{ .tree = self, .want = want };
    }

    /// Forward iterator over bits in one state.
    pub const SkipIterator = struct {
        /// Backing stateful cursor doing the actual traversal.
        inner: SkipCursor,

        /// Returns the next matching bit and advances past it.
        /// - `self` mutated iterator.
        ///
        /// Return: next matching bit index, or null when iteration is complete.
        pub fn next(self: *SkipIterator) ?u32 {
            return self.inner.step();
        }
    };

    /// Creates a forward iterator over bits in the wanted state.
    /// - `self` inspected tree.
    /// - `want` bit state produced by the iterator.
    ///
    /// Return: iterator positioned before the first matching bit.
    pub fn iterator(self: *const Self, want: BitState) SkipIterator {
        return .{ .inner = self.cursor(want) };
    }
};

test "empty tree" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try std.testing.expectEqual(@as(u32, 0), tree.totalBitsCount());
    try std.testing.expectEqual(@as(u32, 0), tree.count(.active));
    try std.testing.expectEqual(@as(u32, 0), tree.count(.inactive));
    try std.testing.expect(tree.next(null, .active) == null);
    try std.testing.expect(tree.next(null, .inactive) == null);
    try std.testing.expect(tree.next(0, .active) == null);
}

test "set/get/next small, exclusive semantics" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 100, .inactive);

    tree.set(3, .active);
    tree.set(5, .active);
    tree.set(64, .active);
    tree.set(65, .active);
    tree.set(5, .active);

    try std.testing.expectEqual(BitState.active, tree.get(3));
    try std.testing.expectEqual(BitState.inactive, tree.get(4));
    try std.testing.expectEqual(BitState.active, tree.get(65));
    try std.testing.expectEqual(@as(u32, 4), tree.count(.active));
    try std.testing.expectEqual(@as(u32, 96), tree.count(.inactive));

    try std.testing.expectEqual(@as(?u32, 3), tree.next(null, .active));
    try std.testing.expectEqual(@as(?u32, 5), tree.next(3, .active));
    try std.testing.expectEqual(@as(?u32, 5), tree.next(4, .active));
    try std.testing.expectEqual(@as(?u32, 64), tree.next(5, .active));
    try std.testing.expectEqual(@as(?u32, 65), tree.next(64, .active));
    try std.testing.expect(tree.next(65, .active) == null);
    try std.testing.expect(tree.next(1000, .active) == null);

    try std.testing.expectEqual(@as(?u32, 0), tree.next(null, .inactive));
    try std.testing.expectEqual(@as(?u32, 1), tree.next(0, .inactive));
    try std.testing.expectEqual(@as(?u32, 4), tree.next(3, .inactive));
    try std.testing.expectEqual(@as(?u32, 66), tree.next(65, .inactive));
    try std.testing.expect(tree.next(99, .inactive) == null);

    tree.set(3, .inactive);
    try std.testing.expectEqual(@as(?u32, 5), tree.next(null, .active));
    try std.testing.expectEqual(@as(u32, 3), tree.count(.active));
}

test "resize precise grow/shrink" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);

    try tree.resize(alloc, 150, .inactive);
    try std.testing.expectEqual(@as(u32, 150), tree.totalBitsCount());
    try std.testing.expect(tree.next(null, .active) == null);
    try std.testing.expectEqual(@as(?u32, 0), tree.next(null, .inactive));

    tree.set(10, .active);
    tree.set(149, .active);

    try tree.resize(alloc, 200, .active);
    try std.testing.expectEqual(BitState.active, tree.get(10));
    try std.testing.expectEqual(BitState.active, tree.get(149));
    try std.testing.expectEqual(BitState.active, tree.get(150));
    try std.testing.expectEqual(BitState.active, tree.get(199));
    try std.testing.expectEqual(@as(u32, 2 + 50), tree.count(.active));
    try std.testing.expectEqual(@as(?u32, 150), tree.next(149, .active));

    try tree.resize(alloc, 11, .inactive);
    try std.testing.expectEqual(@as(u32, 11), tree.totalBitsCount());
    try std.testing.expectEqual(@as(u32, 1), tree.count(.active));
    try std.testing.expectEqual(BitState.active, tree.get(10));
    try std.testing.expect(tree.next(10, .active) == null);
    try std.testing.expectEqual(@as(?u32, 0), tree.next(null, .inactive));

    try std.testing.expectEqual(tree.levels.items[0].items.len, tree.levels.items[0].capacity);

    try tree.resize(alloc, 0, .inactive);
    try std.testing.expectEqual(@as(u32, 0), tree.totalBitsCount());
    try std.testing.expect(tree.next(null, .active) == null);
    try tree.resize(alloc, 70, .active);
    try std.testing.expectEqual(@as(u32, 70), tree.count(.active));
    try std.testing.expectEqual(@as(?u32, 0), tree.next(null, .active));
    try std.testing.expect(tree.next(null, .inactive) == null);
}

test "resizeRetainingCapacity keeps memory" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);

    try tree.resizeRetainingCapacity(alloc, 10000, .inactive);
    const cap_grown = tree.levels.items[0].capacity;
    try std.testing.expect(cap_grown * 64 >= 10000);

    tree.set(9999, .active);
    try tree.resizeRetainingCapacity(alloc, 100, .inactive);
    try std.testing.expectEqual(@as(u32, 100), tree.totalBitsCount());
    try std.testing.expectEqual(@as(u32, 0), tree.count(.active));
    try std.testing.expectEqual(cap_grown, tree.levels.items[0].capacity);
    try tree.resizeRetainingCapacity(alloc, 5000, .active);
    try std.testing.expectEqual(BitState.active, tree.get(4999));
    try std.testing.expectEqual(BitState.inactive, tree.get(50));
    try std.testing.expectEqual(cap_grown, tree.levels.items[0].capacity);
}

test "setRange and clear across word boundaries" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 300, .inactive);

    tree.setRange(60, 10, .active);
    try std.testing.expectEqual(@as(u32, 10), tree.count(.active));
    try std.testing.expectEqual(@as(?u32, 60), tree.next(null, .active));
    try std.testing.expectEqual(@as(?u32, 61), tree.next(60, .active));
    try std.testing.expect(tree.next(69, .active) == null);

    tree.setRange(0, 300, .active);
    try std.testing.expectEqual(@as(u32, 300), tree.count(.active));
    try std.testing.expect(tree.next(null, .inactive) == null);

    tree.setRange(100, 100, .inactive);
    try std.testing.expectEqual(@as(u32, 200), tree.count(.active));
    try std.testing.expectEqual(@as(?u32, 100), tree.next(null, .inactive));
    try std.testing.expectEqual(@as(?u32, 101), tree.next(100, .inactive));

    tree.clear(.inactive);
    try std.testing.expectEqual(@as(u32, 0), tree.count(.active));
    try std.testing.expect(tree.next(null, .active) == null);

    tree.clear(.active);
    try std.testing.expectEqual(@as(u32, 300), tree.count(.active));
    try std.testing.expect(tree.next(null, .inactive) == null);
}

test "iterator matches next loop" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 500, .inactive);
    tree.set(0, .active);
    tree.set(63, .active);
    tree.set(64, .active);
    tree.set(499, .active);

    const expected = [_]u32{ 0, 63, 64, 499 };
    var it = tree.iterator(.active);
    var i: usize = 0;
    while (it.next()) |b| {
        try std.testing.expect(i < expected.len);
        try std.testing.expectEqual(expected[i], b);
        i += 1;
    }
    try std.testing.expectEqual(expected.len, i);
}

test "large sparse skip" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 200_000, .inactive);
    tree.set(1, .active);
    tree.set(100_000, .active);
    tree.set(199_999, .active);
    tree.setRange(50_000, 64, .active);

    try std.testing.expectEqual(@as(?u32, 1), tree.next(null, .active));
    try std.testing.expectEqual(@as(?u32, 50_000), tree.next(1, .active));
    try std.testing.expectEqual(@as(?u32, 50_001), tree.next(50_000, .active));
    try std.testing.expectEqual(@as(?u32, 100_000), tree.next(50_063, .active));
    try std.testing.expectEqual(@as(?u32, 199_999), tree.next(100_000, .active));
    try std.testing.expect(tree.next(199_999, .active) == null);
    try std.testing.expectEqual(@as(u32, 3 + 64), tree.count(.active));

    var it = tree.iterator(.active);
    var n: u32 = 0;
    while (it.next()) |_| n += 1;
    try std.testing.expectEqual(@as(u32, 67), n);
}

test "cursor dense run and seek" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 500, .inactive);
    tree.setRange(100, 200, .active);

    var cursor = tree.cursor(.active);
    var expected: u32 = 100;
    while (cursor.step()) |b| {
        try std.testing.expectEqual(expected, b);
        expected += 1;
    }
    try std.testing.expectEqual(@as(u32, 300), expected);

    cursor.seek(150);
    try std.testing.expectEqual(@as(?u32, 150), cursor.step());
    try std.testing.expectEqual(@as(?u32, 151), cursor.step());

    cursor.seek(299);
    try std.testing.expectEqual(@as(?u32, 299), cursor.step());
    try std.testing.expect(cursor.step() == null);

    cursor.seek(0);
    try std.testing.expectEqual(@as(?u32, 100), cursor.step());

    cursor.seek(500);
    try std.testing.expect(cursor.step() == null);

    var inactive_cursor = tree.cursor(.inactive);
    try std.testing.expectEqual(@as(?u32, 0), inactive_cursor.step());
    inactive_cursor.seek(99);
    try std.testing.expectEqual(@as(?u32, 99), inactive_cursor.step());
    try std.testing.expectEqual(@as(?u32, 300), inactive_cursor.step());
}

test "false positive descent backtracks past long inactive tail" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 200_000, .inactive);
    tree.set(0, .active);

    try std.testing.expect(tree.next(0, .active) == null);
    try std.testing.expect(tree.next(1, .active) == null);
    try std.testing.expectEqual(@as(?u32, 0), tree.next(null, .active));
    var cursor = tree.cursor(.active);
    cursor.seek(1);
    try std.testing.expect(cursor.step() == null);

    tree.set(199_999, .active);
    try std.testing.expectEqual(@as(?u32, 199_999), tree.next(0, .active));
    var tail_cursor = tree.cursor(.active);
    tail_cursor.seek(1);
    try std.testing.expectEqual(@as(?u32, 199_999), tail_cursor.step());
    try std.testing.expect(tail_cursor.step() == null);
}

test "uniform runs emit without leaf reads" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 5000, .inactive);
    tree.setRange(1000, 3000, .active);

    var cursor = tree.cursor(.active);
    var expected: u32 = 1000;
    while (cursor.step()) |b| {
        try std.testing.expectEqual(expected, b);
        expected += 1;
    }
    try std.testing.expectEqual(@as(u32, 4000), expected);

    cursor.seek(2500);
    try std.testing.expectEqual(@as(?u32, 2500), cursor.step());
    try std.testing.expectEqual(@as(?u32, 2501), cursor.step());

    var inactive_cursor = tree.cursor(.inactive);
    try std.testing.expectEqual(@as(?u32, 0), inactive_cursor.step());
    try std.testing.expectEqual(@as(?u32, 1), inactive_cursor.step());
    inactive_cursor.seek(999);
    try std.testing.expectEqual(@as(?u32, 999), inactive_cursor.step());
    try std.testing.expectEqual(@as(?u32, 4000), inactive_cursor.step());
    inactive_cursor.seek(4999);
    try std.testing.expectEqual(@as(?u32, 4999), inactive_cursor.step());
    try std.testing.expect(inactive_cursor.step() == null);
}

const XorShift = struct {
    /// Current generator state. Any nonzero seed starts a full-period cycle.
    s: u64,

    /// Advances the generator and returns the next pseudo-random word.
    /// - `self` mutated generator.
    ///
    /// Return: next pseudo-random 64-bit word.
    fn next(self: *@This()) u64 {
        var x = self.s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.s = x;
        return x;
    }

    /// Returns a pseudo-random value below an exclusive bound.
    /// - `self` mutated generator.
    /// - `n` exclusive upper bound. Zero yields zero.
    ///
    /// Return: value in the range from zero inclusive to the bound exclusive.
    fn below(self: *@This(), n: u32) u32 {
        if (n == 0) return 0;
        return @intCast(self.next() % n);
    }
};

/// Reference exclusive search over the naive boolean model used by the fuzz test.
/// - `model` naive bit values indexed by bit position.
/// - `len` number of valid model entries.
/// - `start` bit index to search after, or null to start from the beginning.
/// - `want` wanted bit value.
///
/// Return: index of the first matching bit, or null when there is none.
fn modelNext(model: []const bool, len: u32, start: ?u32, want: bool) ?u32 {
    var pos: u64 = 0;
    if (start) |s| {
        pos = @as(u64, s) + 1;
        if (pos >= len) return null;
    }
    var i: u32 = @intCast(pos);
    while (i < len) : (i += 1) {
        if (model[i] == want) return i;
    }
    return null;
}

test "fuzz vs naive model" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);

    var model: [4096]bool = [_]bool{false} ** 4096;
    var model_len: u32 = 0;
    var rng = XorShift{ .s = 0x1234_5678_9ABC_DEF1 };
    var precise_toggle = false;

    var step: u32 = 0;
    while (step < 3000) : (step += 1) {
        const op = rng.below(100);
        if (op < 40 and model_len > 0) {
            const b = rng.below(model_len);
            const st: BitState = if (rng.below(2) == 0) .inactive else .active;
            tree.set(b, st);
            model[b] = st == .active;
        } else if (op < 55 and model_len > 0) {
            const s = rng.below(model_len);
            const max_l: u32 = @min(model_len - s, 200);
            const ln = rng.below(max_l + 1);
            const st: BitState = if (rng.below(2) == 0) .inactive else .active;
            tree.setRange(s, ln, st);
            var i: u32 = 0;
            while (i < ln) : (i += 1) model[s + i] = st == .active;
        } else if (op < 65) {
            const want: BitState = if (rng.below(2) == 0) .inactive else .active;
            const start: ?u32 = if (model_len == 0 or rng.below(4) == 0) null else rng.below(model_len);
            const got = tree.next(start, want);
            const want_b = want == .active;
            try std.testing.expectEqual(modelNext(&model, model_len, start, want_b), got);
            var seek_cursor = tree.cursor(want);
            if (start) |s| seek_cursor.seek(@as(u64, s) + 1);
            try std.testing.expectEqual(got, seek_cursor.step());
        } else if (op < 80) {
            const new_len = rng.below(4097);
            const st: BitState = if (rng.below(2) == 0) .inactive else .active;
            precise_toggle = !precise_toggle;
            if (precise_toggle) {
                try tree.resize(alloc, new_len, st);
            } else {
                try tree.resizeRetainingCapacity(alloc, new_len, st);
            }
            if (new_len > model_len) {
                var i: u32 = model_len;
                while (i < new_len) : (i += 1) model[i] = st == .active;
            }
            model_len = new_len;
            try std.testing.expectEqual(model_len, tree.totalBitsCount());
        } else if (model_len > 0) {
            const b = rng.below(model_len);
            try std.testing.expectEqual(model[b], tree.get(b) == .active);
            var c: u32 = 0;
            var i: u32 = 0;
            while (i < model_len) : (i += 1) if (model[i]) {
                c += 1;
            };
            try std.testing.expectEqual(c, tree.count(.active));
            try std.testing.expectEqual(model_len - c, tree.count(.inactive));
        }
        var c: u32 = 0;
        var i: u32 = 0;
        while (i < model_len) : (i += 1) if (model[i]) {
            c += 1;
        };
        try std.testing.expectEqual(c, tree.count(.active));
    }
}

test "tree sparseFactor median gaps" {
    const alloc = std.testing.allocator;
    var tree = BitTree.empty;
    defer tree.deinit(alloc);
    try tree.resize(alloc, 200, .inactive);

    try std.testing.expectEqual(@as(u32, 0), try tree.sparseFactor(alloc, .active));
    tree.set(5, .active);
    try std.testing.expectEqual(@as(u32, 0), try tree.sparseFactor(alloc, .active));
    tree.set(5, .inactive);

    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        tree.set(i * 10, .active);
    }
    try std.testing.expectEqual(@as(u32, 10), try tree.sparseFactor(alloc, .active));

    var mixed = BitTree.empty;
    defer mixed.deinit(alloc);
    try mixed.resize(alloc, 100, .inactive);
    mixed.set(0, .active);
    mixed.set(5, .active);
    mixed.set(6, .active);
    mixed.set(20, .active);
    try std.testing.expectEqual(@as(u32, 5), try mixed.sparseFactor(alloc, .active));

    var even = BitTree.empty;
    defer even.deinit(alloc);
    try even.resize(alloc, 100, .inactive);
    even.set(0, .active);
    even.set(1, .active);
    even.set(4, .active);
    even.set(9, .active);
    even.set(16, .active);
    try std.testing.expectEqual(@as(u32, 3), try even.sparseFactor(alloc, .active));

    var mostly_empty = BitTree.empty;
    defer mostly_empty.deinit(alloc);
    try mostly_empty.resize(alloc, 100, .inactive);
    mostly_empty.set(10, .active);
    mostly_empty.set(20, .active);
    try std.testing.expectEqual(@as(u32, 1), try mostly_empty.sparseFactor(alloc, .inactive));
}

/// Bits per word. One u64 register per word keeps ctz and popcount single-tick.
pub const word_bits: u32 = 64;

/// Median consecutive-bit gap, in bits, at or above which the hierarchy scan wins.
/// Measured crossover inside (1024, 16384]: flat scan wins 3-6x at gaps of 1024
/// and below, the tree wins 2.5x and more at gaps of 16384 and above,
/// stable across 1M and 10M bit sets. Cost-model break-even sits near 6k bits.
/// Threshold placed at 4096: worst-case loss from misplacement stays near 1.3x either way.
pub const treeThresholdGapBits: u32 = 4096;

/// Scan strategy selected from a measured sparseness factor.
pub const ScanKind = enum {
    /// Direct linear word scan. Best for small to moderate gaps without long uniform runs.
    linear,
    /// Hierarchy-assisted scan. Emits uniform runs by counter, skips huge uniform gaps.
    tree,
};

/// Selects the faster scan strategy for a measured median gap.
/// Zero means fewer than two set bits: the tree answers by a root check
/// instead of a full linear walk. Gaps at or above the sparse threshold
/// go tree: huge gaps are skipped by summaries. Anything in between,
/// including fully dense data, scans linearly: the flat loop wins there
/// or ties within a robust, layout-independent margin.
/// - `median_gap` median consecutive-bit gap in bits, as reported by sparseFactor.
///
/// Return: recommended scan strategy.
pub fn chooseScan(median_gap: u32) ScanKind {
    if (median_gap == 0) return .tree;
    if (median_gap >= treeThresholdGapBits) return .tree;
    return .linear;
}

/// Counts words needed to hold the given bit count.
/// - `bits` total bit count.
///
/// Return: number of u64 words.
inline fn wordsFor(bits: u32) usize {
    return std.math.divCeil(u32, bits, word_bits) catch unreachable;
}

/// Builds the valid-bit mask for the last word.
/// Tail bits past the total count are always zero.
/// - `total_bits` total bit count.
///
/// Return: mask with one bit per valid tail bit.
inline fn lastWordMask(total_bits: u32) u64 {
    const r: u6 = @intCast(total_bits & 63);
    if (r == 0) return all_ones;
    return (@as(u64, 1) << r) - 1;
}

/// Builds the masked content of one word for a scan in the wanted state.
/// Bits before the skip offset and tail bits past the total count are cleared.
/// Specialized at compile time, so the polarity branch vanishes from the scan loop.
/// - `words` backing word buffer.
/// - `total` total bit count.
/// - `idx` scanned word index.
/// - `skip` number of leading word bits to clear.
/// - `want` wanted bit state.
///
/// Return: word with only candidate bits set.
inline fn maskedFlatWord(words: []const u64, total: u32, idx: usize, skip: u6, comptime want: BitState) u64 {
    var w: u64 = words[idx];
    if (want == .inactive) w = ~w;
    if (skip != 0) w &= all_ones << skip;
    if (idx + 1 == words.len and (total & 63) != 0) {
        w &= lastWordMask(total);
    }
    return w;
}

/// Flat bitset: dense u64 words plus an active counter, no hierarchy at all.
/// Iteration is a linear word scan with ctz per matching word.
pub const FlatBitSet = struct {
    /// Short alias of the enclosing type.
    const Self = @This();

    /// Backing words, 64 bits each.
    words: ListA64(u64) = .empty,
    /// Number of addressable bits.
    total_bits: u32 = 0,
    /// Number of set bits. Updated by every mutating operation.
    active_count: u32 = 0,

    /// Empty bitset with no bits and no allocated words.
    pub const empty: Self = .{};

    /// Releases the word buffer and invalidates the bitset.
    /// - `self` bitset to destroy.
    /// - `allocator` allocator that owns the word buffer.
    pub fn deinit(self: *Self, allocator: Allocator) void {
        self.words.deinit(allocator);
        self.* = undefined;
    }

    /// Reports the number of addressable bits.
    /// - `self` inspected bitset.
    ///
    /// Return: total bit count.
    pub inline fn totalBitsCount(self: *const Self) u32 {
        return self.total_bits;
    }

    /// Counts bits in the given state in O(1) through the maintained counter.
    /// - `self` inspected bitset.
    /// - `state` bit state to count.
    ///
    /// Return: number of bits in the requested state.
    pub inline fn count(self: *const Self, state: BitState) u32 {
        return switch (state) {
            .active => self.active_count,
            .inactive => self.total_bits - self.active_count,
        };
    }

    /// Computes the sparseness factor of bits in the wanted state.
    /// Defined as the median gap between consecutive set bits, measured in bits.
    /// Returns zero when fewer than two bits are set. Manual call by design:
    /// the value is not maintained incrementally, recompute it after bulk updates.
    /// - `self` inspected bitset.
    /// - `allocator` allocator for the temporary gap buffer, freed before return.
    /// - `want` bit state to measure, compile-time known.
    ///
    /// Return: median consecutive-bit gap in bits, or zero when undefined.
    pub fn sparseFactor(self: *const Self, allocator: Allocator, comptime want: BitState) Allocator.Error!u32 {
        const found: u32 = self.count(want);
        if (found < 2) return 0;
        const gaps: []u32 = try allocator.alloc(u32, @as(usize, @intCast(found - 1)));
        defer allocator.free(gaps);
        var it = self.cursor(want);
        var prev: u32 = it.step() orelse return 0;
        var n: usize = 0;
        while (it.step()) |b| {
            gaps[n] = b - prev;
            prev = b;
            n += 1;
        }
        std.debug.assert(n == gaps.len);
        std.mem.sortUnstable(u32, gaps, {}, std.sort.asc(u32));
        return gaps[(gaps.len - 1) / 2];
    }

    /// Reads a single bit.
    /// - `self` inspected bitset.
    /// - `bit` bit index, must be below the total bit count.
    ///
    /// Return: state of the requested bit.
    pub fn get(self: *const Self, bit: u32) BitState {
        std.debug.assert(bit < self.total_bits);
        const w: u64 = self.words.items[bit >> 6];
        const s: u6 = @intCast(bit & 63);
        return if ((w >> s) & 1 == 1) .active else .inactive;
    }

    /// Sets a single bit.
    /// - `self` mutated bitset.
    /// - `bit` bit index, must be below the total bit count.
    /// - `state` state to store.
    pub fn set(self: *Self, bit: u32, state: BitState) void {
        std.debug.assert(bit < self.total_bits);
        const word_idx: usize = bit >> 6;
        const shift: u6 = @intCast(bit & 63);
        const mask: u64 = @as(u64, 1) << shift;
        const word: *u64 = &self.words.items[word_idx];
        const was_active = (word.* & mask) != 0;
        const want_active = state == .active;
        if (was_active == want_active) return;
        if (want_active) {
            word.* |= mask;
            self.active_count += 1;
        } else {
            word.* &= ~mask;
            self.active_count -= 1;
        }
    }

    /// Resizes the bitset, growing capacity geometrically and retaining it on shrink.
    /// New bits are initialized with the requested state.
    /// - `self` mutated bitset.
    /// - `allocator` allocator that owns the word buffer.
    /// - `new_bits_count` requested total bit count.
    /// - `new_bits_state` initial state of bits added by growth.
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
                    const word_start: u64 = @as(u64, tail_idx) * word_bits;
                    const fill_to: u64 = @min(@as(u64, new_bits_count), word_start + word_bits);
                    const lo: u32 = @intCast(@as(u64, old_bits) - word_start);
                    const hi: u32 = @intCast(fill_to - word_start);
                    var m: u64 = all_ones << @as(u6, @intCast(lo));
                    if (hi < word_bits) {
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

    /// Counts set bits inside a bit range.
    /// - `self` inspected bitset.
    /// - `start` first bit index of the range.
    /// - `len` number of covered bits.
    ///
    /// Return: number of set bits inside the range.
    fn countRangeActive(self: *const Self, start: u32, len: u32) u32 {
        if (len == 0) return 0;
        const end: u64 = @as(u64, start) + len;
        var w: usize = @intCast(start >> 6);
        const last_word: usize = @intCast((end - 1) >> 6);
        var acc: u32 = 0;
        while (w <= last_word) : (w += 1) {
            const ws: u64 = @as(u64, w) * word_bits;
            var m: u64 = all_ones;
            if (@as(u64, start) > ws) {
                const lo: u6 = @intCast(@as(u64, start) - ws);
                m &= all_ones << lo;
            }
            if (end - ws < word_bits) {
                const hi: u6 = @intCast(end - ws);
                m &= (@as(u64, 1) << hi) - 1;
            }
            acc += @intCast(@popCount(self.words.items[w] & m));
        }
        return acc;
    }

    /// Stateful forward cursor over bits in one state. Primary query path for full scans.
    /// Caches the masked content of the current word, so consecutive bits inside one
    /// word cost a single ctz plus a bit clear each. Refill scans words linearly.
    /// Specialized at compile time for the wanted state: no polarity branches at runtime.
    /// - `want` wanted bit state, compile-time known.
    ///
    /// Return: cursor type producing bits in the requested state.
    pub fn LinearCursor(comptime want: BitState) type {
        return struct {
        /// Inspected bitset. Must outlive the cursor and must not be mutated during iteration.
        flat: *const FlatBitSet,
        /// Bit position where the next step starts examining. Inclusive.
        pos: u64 = 0,
        /// Not-yet-returned candidate bits of the cached word. Meaningful only with `has_cache`.
        bits: u64 = 0,
        /// Word index backing `bits`.
        cached_word: usize = 0,
        /// Whether `bits` and `cached_word` hold a usable cached word.
        has_cache: bool = false,

        /// Moves the examination position. Invalidates the cached word.
        /// Monotonic forward stepping without seek calls is the fast path.
        /// - `self` mutated cursor.
        /// - `pos` bit position where examination resumes. Inclusive.
        pub fn seek(self: *@This(), pos: u64) void {
            self.pos = pos;
            self.has_cache = false;
        }

        /// Returns the next matching bit at or after the examination position.
        /// Advances past the returned bit, clearing it from the cached word, so the
        /// following call resumes inside the same word whenever bits are left.
        /// Bits inside the cached word are served without touching further words.
        /// - `self` mutated cursor.
        ///
        /// Return: next matching bit index, or null when iteration is complete.
        pub inline fn step(self: *@This()) ?u32 {
            const flat = self.flat;
            const total = flat.total_bits;
            if (total == 0) return null;
            while (self.pos < total) {
                if (self.has_cache and (self.pos >> 6) == @as(u64, self.cached_word)) {
                    if (self.bits == 0) {
                        self.pos = (@as(u64, self.cached_word) + 1) * @as(u64, word_bits);
                        self.has_cache = false;
                        continue;
                    }
                    return self.takeCachedBit(total);
                }
                self.has_cache = false;
                const lw: usize = self.refill() orelse return null;
                const base: u64 = @as(u64, lw) * @as(u64, word_bits);
                if (self.pos < base) self.pos = base;
            }
            return null;
        }

        /// Scans words linearly from the examination position for the next word
        /// holding a candidate bit, caches its masked content and returns its index.
        /// - `self` mutated cursor.
        ///
        /// Return: cached word index, or null when no word below can match.
        inline fn refill(self: *@This()) ?usize {
            return self.refillImpl();
        }

        /// Linear refill over the backing words.
        /// The first partial word is scanned with the skip mask, steady words plainly.
        /// - `self` mutated cursor.
        ///
        /// Return: cached word index, or null when no word below can match.
        fn refillImpl(self: *@This()) ?usize {
            const words = self.flat.words.items;
            const total = self.flat.total_bits;
            var i: usize = @intCast(self.pos / word_bits);
            if (i < words.len) {
                const skip: u6 = @intCast(self.pos - @as(u64, i) * word_bits);
                const w: u64 = maskedFlatWord(words, total, i, skip, want);
                if (w != 0) {
                    self.bits = w;
                    self.cached_word = i;
                    self.has_cache = true;
                    return i;
                }
                i += 1;
            }
            while (i < words.len) : (i += 1) {
                const w: u64 = maskedFlatWord(words, total, i, 0, want);
                if (w != 0) {
                    self.bits = w;
                    self.cached_word = i;
                    self.has_cache = true;
                    return i;
                }
            }
            return null;
        }

        /// Returns the lowest set bit of the cached word and clears it from the cache.
        /// The caller guarantees a warm cache positioned inside the cached word.
        /// - `self` mutated cursor.
        /// - `total` total bit count bounding the result.
        ///
        /// Return: next matching bit index, or null on a defeated tail guard.
        inline fn takeCachedBit(self: *@This(), total: u32) ?u32 {
            const b: u32 = @intCast(@as(u64, self.cached_word) * @as(u64, word_bits) + @as(u64, @ctz(self.bits)));
            if (b >= total) return null;
            self.bits &= self.bits - 1;
            if (self.bits == 0) {
                self.pos = (@as(u64, self.cached_word) + 1) * @as(u64, word_bits);
                self.has_cache = false;
            } else {
                self.pos = @as(u64, b) + 1;
            }
            return b;
        }
        };
    }

    /// Creates a forward cursor over bits in the wanted state, positioned at bit zero.
    /// - `self` inspected bitset.
    /// - `want` bit state produced by the cursor, compile-time known.
    ///
    /// Return: cursor positioned before the first matching bit.
    pub fn cursor(self: *const Self, comptime want: BitState) LinearCursor(want) {
        return .{ .flat = self };
    }

    /// Forward iterator over bits in one state.
    /// Specialized at compile time for the wanted state, like its cursor.
    /// - `want` wanted bit state, compile-time known.
    ///
    /// Return: iterator type producing bits in the requested state.
    pub fn LinearIterator(comptime want: BitState) type {
        return struct {
        /// Backing stateful cursor doing the actual traversal.
        inner: LinearCursor(want),

        /// Returns the next matching bit and advances past it.
        /// - `self` mutated iterator.
        ///
        /// Return: next matching bit index, or null when iteration is complete.
        pub fn next(self: *@This()) ?u32 {
            return self.inner.step();
        }
        };
    }

    /// Creates a forward iterator over bits in the wanted state.
    /// - `self` inspected bitset.
    /// - `want` bit state produced by the iterator, compile-time known.
    ///
    /// Return: iterator positioned before the first matching bit.
    pub fn iterator(self: *const Self, comptime want: BitState) LinearIterator(want) {
        return .{ .inner = self.cursor(want) };
    }
};

test "flat set/get/cursor small" {
    const alloc = std.testing.allocator;
    var bits = FlatBitSet.empty;
    defer bits.deinit(alloc);
    try bits.resize(alloc, 100, .inactive);

    bits.set(3, .active);
    bits.set(5, .active);
    bits.set(64, .active);
    bits.set(65, .active);
    bits.set(5, .active);

    try std.testing.expectEqual(BitState.active, bits.get(3));
    try std.testing.expectEqual(BitState.inactive, bits.get(4));
    try std.testing.expectEqual(@as(u32, 4), bits.count(.active));
    try std.testing.expectEqual(@as(u32, 96), bits.count(.inactive));

    var cursor = bits.cursor(.active);
    try std.testing.expectEqual(@as(?u32, 3), cursor.step());
    try std.testing.expectEqual(@as(?u32, 5), cursor.step());
    try std.testing.expectEqual(@as(?u32, 64), cursor.step());
    try std.testing.expectEqual(@as(?u32, 65), cursor.step());
    try std.testing.expect(cursor.step() == null);

    cursor.seek(4);
    try std.testing.expectEqual(@as(?u32, 5), cursor.step());
    cursor.seek(5);
    try std.testing.expectEqual(@as(?u32, 5), cursor.step());
    cursor.seek(6);
    try std.testing.expectEqual(@as(?u32, 64), cursor.step());
    cursor.seek(65);
    try std.testing.expectEqual(@as(?u32, 65), cursor.step());
    cursor.seek(66);
    try std.testing.expect(cursor.step() == null);
    cursor.seek(1000);
    try std.testing.expect(cursor.step() == null);

    var inactive_cursor = bits.cursor(.inactive);
    try std.testing.expectEqual(@as(?u32, 0), inactive_cursor.step());
    inactive_cursor.seek(3);
    try std.testing.expectEqual(@as(?u32, 4), inactive_cursor.step());
    inactive_cursor.seek(65);
    try std.testing.expectEqual(@as(?u32, 66), inactive_cursor.step());
    inactive_cursor.seek(99);
    try std.testing.expectEqual(@as(?u32, 99), inactive_cursor.step());
    try std.testing.expect(inactive_cursor.step() == null);

    bits.set(3, .inactive);
    var after = bits.cursor(.active);
    try std.testing.expectEqual(@as(?u32, 5), after.step());
    try std.testing.expectEqual(@as(u32, 3), bits.count(.active));
}

test "flat resize grow/shrink with fill states" {
    const alloc = std.testing.allocator;
    var bits = FlatBitSet.empty;
    defer bits.deinit(alloc);

    try bits.resize(alloc, 150, .inactive);
    try std.testing.expectEqual(@as(u32, 150), bits.totalBitsCount());
    var empty_cursor = bits.cursor(.active);
    try std.testing.expect(empty_cursor.step() == null);

    bits.set(10, .active);
    bits.set(149, .active);

    try bits.resize(alloc, 200, .active);
    try std.testing.expectEqual(BitState.active, bits.get(10));
    try std.testing.expectEqual(BitState.active, bits.get(149));
    try std.testing.expectEqual(BitState.active, bits.get(150));
    try std.testing.expectEqual(BitState.active, bits.get(199));
    try std.testing.expectEqual(@as(u32, 52), bits.count(.active));

    try bits.resize(alloc, 11, .inactive);
    try std.testing.expectEqual(@as(u32, 11), bits.totalBitsCount());
    try std.testing.expectEqual(@as(u32, 1), bits.count(.active));
    var tail_cursor = bits.cursor(.active);
    tail_cursor.seek(11);
    try std.testing.expect(tail_cursor.step() == null);

    try bits.resize(alloc, 0, .inactive);
    try std.testing.expectEqual(@as(u32, 0), bits.totalBitsCount());
    var zero_cursor = bits.cursor(.active);
    try std.testing.expect(zero_cursor.step() == null);
    try bits.resize(alloc, 70, .active);
    try std.testing.expectEqual(@as(u32, 70), bits.count(.active));
    var full_cursor = bits.cursor(.inactive);
    try std.testing.expect(full_cursor.step() == null);
}

test "flat cursor dense run and seek" {
    const alloc = std.testing.allocator;
    var bits = FlatBitSet.empty;
    defer bits.deinit(alloc);
    try bits.resize(alloc, 500, .inactive);
    var b: u32 = 100;
    while (b < 300) : (b += 1) {
        bits.set(b, .active);
    }

    var cursor = bits.cursor(.active);
    var expected: u32 = 100;
    while (cursor.step()) |v| {
        try std.testing.expectEqual(expected, v);
        expected += 1;
    }
    try std.testing.expectEqual(@as(u32, 300), expected);

    cursor.seek(150);
    try std.testing.expectEqual(@as(?u32, 150), cursor.step());
    cursor.seek(299);
    try std.testing.expectEqual(@as(?u32, 299), cursor.step());
    try std.testing.expect(cursor.step() == null);
    cursor.seek(0);
    try std.testing.expectEqual(@as(?u32, 100), cursor.step());
    cursor.seek(500);
    try std.testing.expect(cursor.step() == null);
}

test "flat sparseFactor median gaps" {
    const alloc = std.testing.allocator;
    var bits = FlatBitSet.empty;
    defer bits.deinit(alloc);
    try bits.resize(alloc, 200, .inactive);

    try std.testing.expectEqual(@as(u32, 0), try bits.sparseFactor(alloc, .active));
    bits.set(5, .active);
    try std.testing.expectEqual(@as(u32, 0), try bits.sparseFactor(alloc, .active));
    bits.set(5, .inactive);

    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        bits.set(i * 10, .active);
    }
    try std.testing.expectEqual(@as(u32, 10), try bits.sparseFactor(alloc, .active));

    var mixed = FlatBitSet.empty;
    defer mixed.deinit(alloc);
    try mixed.resize(alloc, 100, .inactive);
    mixed.set(0, .active);
    mixed.set(5, .active);
    mixed.set(6, .active);
    mixed.set(20, .active);
    try std.testing.expectEqual(@as(u32, 5), try mixed.sparseFactor(alloc, .active));

    var even = FlatBitSet.empty;
    defer even.deinit(alloc);
    try even.resize(alloc, 100, .inactive);
    even.set(0, .active);
    even.set(1, .active);
    even.set(4, .active);
    even.set(9, .active);
    even.set(16, .active);
    try std.testing.expectEqual(@as(u32, 3), try even.sparseFactor(alloc, .active));

    var mostly_empty = FlatBitSet.empty;
    defer mostly_empty.deinit(alloc);
    try mostly_empty.resize(alloc, 100, .inactive);
    mostly_empty.set(10, .active);
    mostly_empty.set(20, .active);
    try std.testing.expectEqual(@as(u32, 1), try mostly_empty.sparseFactor(alloc, .inactive));
}

test "chooseScan follows the measured threshold" {
    try std.testing.expectEqual(ScanKind.tree, chooseScan(0));
    try std.testing.expectEqual(ScanKind.linear, chooseScan(1));
    try std.testing.expectEqual(ScanKind.linear, chooseScan(2));
    try std.testing.expectEqual(ScanKind.linear, chooseScan(1023));
    try std.testing.expectEqual(ScanKind.linear, chooseScan(treeThresholdGapBits - 1));
    try std.testing.expectEqual(ScanKind.tree, chooseScan(treeThresholdGapBits));
    try std.testing.expectEqual(ScanKind.tree, chooseScan(std.math.maxInt(u32)));
}

test "detectScanKind decides from coarse structure" {
    const alloc = std.testing.allocator;

    var empty_tree = BitTree.empty;
    defer empty_tree.deinit(alloc);
    try std.testing.expectEqual(ScanKind.linear, empty_tree.detectScanKind(.active));

    var tiny = BitTree.empty;
    defer tiny.deinit(alloc);
    try tiny.resize(alloc, 64, .active);
    try std.testing.expectEqual(ScanKind.linear, tiny.detectScanKind(.active));

    var dense = BitTree.empty;
    defer dense.deinit(alloc);
    try dense.resize(alloc, 10_000, .active);
    try std.testing.expectEqual(ScanKind.linear, dense.detectScanKind(.active));

    var cleared = BitTree.empty;
    defer cleared.deinit(alloc);
    try cleared.resize(alloc, 10_000, .inactive);
    try std.testing.expectEqual(ScanKind.tree, cleared.detectScanKind(.active));

    var half = BitTree.empty;
    defer half.deinit(alloc);
    try half.resize(alloc, 10_000, .inactive);
    half.setRange(0, 5000, .active);
    try std.testing.expectEqual(ScanKind.tree, half.detectScanKind(.active));

    var ultra = BitTree.empty;
    defer ultra.deinit(alloc);
    try ultra.resize(alloc, 200_000, .inactive);
    ultra.set(7, .active);
    ultra.set(199_993, .active);
    try std.testing.expectEqual(ScanKind.tree, ultra.detectScanKind(.active));

    var scattered = BitTree.empty;
    defer scattered.deinit(alloc);
    try scattered.resize(alloc, 4096, .inactive);
    var rng = XorShift{ .s = 0x2545_F491_4F6C_DD1D };
    var i: u32 = 0;
    while (i < 4096) : (i += 1) {
        if (rng.below(2) == 0) scattered.set(i, .active);
    }
    try std.testing.expectEqual(ScanKind.linear, scattered.detectScanKind(.active));

    var strided = BitTree.empty;
    defer strided.deinit(alloc);
    try strided.resize(alloc, 4096, .inactive);
    var b: u32 = 0;
    while (b < 4096) : (b += 3) {
        strided.set(b, .active);
    }
    try std.testing.expectEqual(ScanKind.linear, strided.detectScanKind(.active));
}

test "detectScanKind agrees with manual rule on stride patterns" {
    const alloc = std.testing.allocator;
    const strides = [_]u32{ 1, 8, 1024, 16384 };
    for (strides) |stride| {
        var tree = BitTree.empty;
        defer tree.deinit(alloc);
        try tree.resize(alloc, 65536, .inactive);
        var bit: u32 = 0;
        while (bit < 65536) : (bit += stride) {
            tree.set(bit, .active);
        }
        const factor: u32 = try tree.sparseFactor(alloc, .active);
        try std.testing.expectEqual(stride, factor);
        try std.testing.expectEqual(chooseScan(factor), tree.detectScanKind(.active));
    }
}
