const std = @import("std");
const math = std.math;
const utilities = @import("utilities.zig");
const bit_word = @import("bit_word.zig");
const Allocator = std.mem.Allocator;
const ListA64 = utilities.ListA64;

const BitState = utilities.BitState;

pub fn BitSet(comptime Word: type) type {
    comptime {
        utilities.assertUnsignedPowerOfTwoInt(Word);
    }

    const bw = bit_word.BitWord(Word);

    return struct {
        const Self = @This();

        words: ListA64(Word) = .empty,
        bits_count: u32 = 0,
        active_bits_counter: u32 = 0,

        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.words.deinit(allocator);
        }

        pub fn resize(self: *Self, allocator: Allocator, new_bits_count: u32, created_bits_value: BitState) !void {
            const old_bits_count = self.bits_count;
            if (old_bits_count == new_bits_count) return;

            self.bits_count = new_bits_count;
            const words = self.words.items;
            const old_words_count: u32 = @truncate(words.len);
            const new_words_count = bw.bitsToWordsCount(new_bits_count);

            if (old_bits_count < new_bits_count) {
                const bits_count_delta = new_bits_count - old_bits_count;
                if (created_bits_value == .active) {
                    self.active_bits_counter += bits_count_delta;
                }

                const created_word_value = created_bits_value.toWordState(Word);
                if (bw.bitIdInWord(old_bits_count) != 0) {
                    const last_old_bit_id = old_bits_count - 1;
                    const last_old_word_id = bw.bitToWordId(last_old_bit_id);
                    const last_old_word = words[last_old_word_id];
                    const end_mask = bw.maskEnd(@truncate(bw.remainBitsInWord(bw.bitIdInWord(last_old_bit_id))));
                    words[last_old_word_id] = bw.merge(last_old_word, created_word_value, end_mask);
                }

                if (new_words_count != old_words_count) {
                    try self.words.resize(allocator, new_words_count);
                    const new_words = self.words.items;

                    for (old_words_count..new_words_count) |i| {
                        new_words[i] = created_word_value;
                    }
                }

                if (bw.bitIdInWord(new_bits_count) != 0) {
                    const last_new_bit_id = new_bits_count - 1;
                    const last_new_word_id = bw.bitToWordId(last_new_bit_id);
                    self.words.items[last_new_word_id] &= bw.maskStart(bw.bitIdInWord(new_bits_count));
                }
            } else {
                if (bw.bitIdInWord(new_bits_count) != 0) {
                    const last_new_bit_id = new_bits_count - 1;
                    const last_old_bit_id = old_bits_count - 1;
                    const last_new_word_id = bw.bitToWordId(last_new_bit_id);
                    const last_old_word_id = bw.bitToWordId(last_old_bit_id);
                    const last_new_word = words[last_new_word_id];
                    if (new_words_count == old_words_count) {
                        const start_end_mask = bw.maskStartEnd(
                            bw.bitIdInWord(new_bits_count),
                            bw.remainBitsInWord(last_old_bit_id),
                        );
                        const new_word = bw.merge(last_new_word, 0, start_end_mask);
                        self.active_bits_counter -= @popCount(new_word);
                    } else {
                        const start_mask = bw.maskStart(bw.bitIdInWord(new_bits_count));
                        const end_mask = bw.maskEnd(bw.remainBitsInWord(last_old_bit_id));
                        const masked_last_new_word = bw.merge(last_new_word, 0, start_mask);
                        const masked_last_old_word = bw.merge(words[last_old_word_id], 0, end_mask);
                        self.active_bits_counter -= (@popCount(masked_last_new_word) + @popCount(masked_last_old_word));
                    }
                    words[last_new_word_id] &= bw.maskStart(bw.bitIdInWord(new_bits_count));
                }

                if (new_words_count != old_words_count) {
                    const skip_last_word: u32 = @intFromBool(bw.bitIdInWord(new_bits_count) != 0);
                    for (new_words_count..old_words_count - skip_last_word) |i| {
                        self.active_bits_counter -= @popCount(words[i]);
                    }

                    try self.words.resize(allocator, new_words_count);
                }
            }
        }
    };
}

pub fn Layer(comptime Word: type) type {
    comptime {
        utilities.assertUnsignedPowerOfTwoInt(Word);
    }

    const bw = bit_word.BitWord(Word);

    return struct {
        const Self = @This();

        activity: ListA64(Word) = .empty,
        mixed: ListA64(Word) = .empty,
        bits_count: u32 = 0,
        state_counters: [4]u32 = .{ 0, 0, 0, 0 },

        pub const State = enum(u2) {
            inactive = 0,
            active = 1,
            mixed = 2,
            deep_mixed = 3,

            pub const inactive_u2: u2 = 0;
            pub const active_u2: u2 = 1;
            pub const mixed_u2: u2 = 2;
            pub const deep_mixed_u2: u2 = 3;

            pub const inactive_u32: u32 = 0;
            pub const active_u32: u32 = 1;
            pub const mixed_u32: u32 = 2;
            pub const deep_mixed_u32: u32 = 3;

            pub inline fn fromBits(activity: BitState, mixed: BitState) State {
                const b0: u2 = @intFromEnum(activity);
                const b1: u2 = @as(u2, @intFromEnum(mixed)) << 1;
                return @enumFromInt(b0 | b1);
            }

            pub inline fn activityBit(self: State) BitState {
                const b0: u1 = @truncate(@as(u2, @intFromEnum(self)));
                return @enumFromInt(b0);
            }

            pub inline fn mixedBit(self: State) BitState {
                const b0: u1 = @truncate(@as(u2, @intFromEnum(self)) >> 1);
                return @enumFromInt(b0);
            }
        };

        pub const StateCounts = struct { inactive: u32, active: u32, mixed: u32, deep: u32 };

        /// Counts the 4 (activity, mixed) states restricted to `mask`.
        inline fn countStates(activity: Word, mixed: Word, mask: Word) StateCounts {
            const a = activity & mask;
            const m = mixed & mask;
            return .{
                .inactive = @popCount(~a & ~m & mask),
                .active = @popCount(a & ~m & mask),
                .mixed = @popCount(~a & m & mask),
                .deep = @popCount(a & m & mask),
            };
        }

        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.activity.deinit(allocator);
            self.mixed.deinit(allocator);
        }

        pub fn setWord(self: *Self, id: u32, activity: Word, mixed: Word, mask: Word) void {
            if (mask == 0) return;
            std.debug.assert(id < self.activity.items.len);
            std.debug.assert(self.activity.items.len == self.mixed.items.len);
            var eff: Word = mask;
            // Ignore padding beyond bits_count in the last word: never write
            // it and never count it.
            const words_count = self.activity.items.len;
            if (words_count > 0 and id == words_count - 1) {
                const used: u32 = bw.bitIdInWord(self.bits_count);
                if (used != 0) eff &= bw.maskStartClamped(used);
            }
            if (eff == 0) return;

            const old_a: Word = self.activity.items[id];
            const old_m: Word = self.mixed.items[id];
            const new_a: Word = bw.merge(old_a, activity, eff);
            const new_m: Word = bw.merge(old_m, mixed, eff);
            if (new_a == old_a and new_m == old_m) return;

            const o = countStates(old_a, old_m, eff);
            const n = countStates(activity, mixed, eff);

            self.activity.items[id] = new_a;
            self.mixed.items[id] = new_m;

            self.state_counters[State.inactive_u32] = self.state_counters[State.inactive_u32] - o.inactive + n.inactive;
            self.state_counters[State.active_u32] = self.state_counters[State.active_u32] - o.active + n.active;
            self.state_counters[State.mixed_u32] = self.state_counters[State.mixed_u32] - o.mixed + n.mixed;
            self.state_counters[State.deep_mixed_u32] = self.state_counters[State.deep_mixed_u32] - o.deep + n.deep;
        }

        pub fn resize(
            self: *Self,
            allocator: Allocator,
            new_bits_count: u32,
            created_bits_value: State,
        ) !void {
            const old_bits_count = self.bits_count;
            if (new_bits_count == old_bits_count) return;
            const old_words_count: usize = bw.bitsToWordsCount(old_bits_count);
            const new_words_count: usize = bw.bitsToWordsCount(new_bits_count);
            const created_a: Word = created_bits_value.activityBit().toWordState(Word);
            const created_m: Word = created_bits_value.mixedBit().toWordState(Word);

            if (new_bits_count > old_bits_count) {
                try self.activity.resize(allocator, new_words_count);
                errdefer self.activity.resize(allocator, old_words_count) catch {};
                try self.mixed.resize(allocator, new_words_count);
                errdefer self.mixed.resize(allocator, old_words_count) catch {};

                var act = self.activity.items;
                var mix = self.mixed.items;

                const old_used: u32 = bw.bitIdInWord(old_bits_count);
                const new_used: u32 = bw.bitIdInWord(new_bits_count);

                if (old_words_count < new_words_count) {
                    if (old_words_count > 0 and old_used != 0) {
                        const old_valid = bw.maskStartClamped(old_used);
                        const idx = old_words_count - 1;
                        act[idx] = bw.merge(created_a, act[idx], old_valid);
                        mix[idx] = bw.merge(created_m, mix[idx], old_valid);
                    }
                    if (new_words_count > old_words_count) {
                        @memset(act[old_words_count..new_words_count], created_a);
                        @memset(mix[old_words_count..new_words_count], created_m);
                    }
                } else {
                    const idx = new_words_count - 1;
                    const old_valid = bw.maskStartClamped(old_used);
                    const new_valid = bw.maskStartClamped(new_used);
                    const range = new_valid & ~old_valid;
                    act[idx] = (act[idx] & old_valid) | (created_a & range);
                    mix[idx] = (mix[idx] & old_valid) | (created_m & range);
                }

                if (new_used != 0 and new_words_count > 0) {
                    const new_valid = bw.maskStartClamped(new_used);
                    act[new_words_count - 1] &= new_valid;
                    mix[new_words_count - 1] &= new_valid;
                }

                self.state_counters[@intFromEnum(created_bits_value)] += new_bits_count - old_bits_count;
                self.bits_count = new_bits_count;
            } else {
                const act_old = self.activity.items;
                const mix_old = self.mixed.items;
                var c_inactive: u32 = 0;
                var c_active: u32 = 0;
                var c_mixed: u32 = 0;
                var c_deep: u32 = 0;
                const new_used: u32 = bw.bitIdInWord(new_bits_count);
                var w: usize = 0;
                if (new_words_count > 0 and new_used != 0) {
                    const rm: Word = ~bw.maskStartClamped(new_used);
                    const c = countStates(act_old[new_words_count - 1], mix_old[new_words_count - 1], rm);
                    c_inactive += c.inactive;
                    c_active += c.active;
                    c_mixed += c.mixed;
                    c_deep += c.deep;
                    w = new_words_count;
                } else {
                    w = new_words_count;
                }
                while (w < old_words_count) : (w += 1) {
                    const a: Word = act_old[w];
                    const m: Word = mix_old[w];
                    var mm: Word = bw.max_value;
                    if (w + 1 == old_words_count) {
                        const old_used: u32 = bw.bitIdInWord(old_bits_count);
                        if (old_used != 0) mm = bw.maskStartClamped(old_used);
                    }
                    const c = countStates(a, m, mm);
                    c_inactive += c.inactive;
                    c_active += c.active;
                    c_mixed += c.mixed;
                    c_deep += c.deep;
                }
                self.state_counters[State.inactive_u32] -|= c_inactive;
                self.state_counters[State.active_u32] -|= c_active;
                self.state_counters[State.mixed_u32] -|= c_mixed;
                self.state_counters[State.deep_mixed_u32] -|= c_deep;

                try self.activity.resize(allocator, new_words_count);
                try self.mixed.resize(allocator, new_words_count);

                if (new_used != 0 and new_words_count > 0) {
                    const new_valid = bw.maskStartClamped(new_used);
                    self.activity.items[new_words_count - 1] &= new_valid;
                    self.mixed.items[new_words_count - 1] &= new_valid;
                }
                self.bits_count = new_bits_count;
            }
        }
    };
}

pub const BitTree = struct {};

const t = std.testing;

fn scanCounters(comptime Word: type, layer: *const Layer(Word)) [4]u32 {
    const BW = bit_word.BitWord(Word);
    var out = [4]u32{ 0, 0, 0, 0 };
    var i: u32 = 0;
    while (i < layer.bits_count) : (i += 1) {
        const bit_word_id = BW.bitToWordId(i);
        const bit_id_in_word = BW.bitIdInWord(i);
        const a = BW.readBit(layer.activity.items[bit_word_id], @truncate(bit_id_in_word));
        const m = BW.readBit(layer.mixed.items[bit_word_id], @truncate(bit_id_in_word));
        out[@intFromEnum(Layer(Word).State.fromBits(a, m))] += 1;
    }
    return out;
}

fn expectLayerValid(comptime Word: type, layer: *const Layer(Word)) !void {
    const BW = bit_word.BitWord(Word);
    const want_words: usize = BW.bitsToWordsCount(layer.bits_count);
    try t.expectEqual(want_words, layer.activity.items.len);
    try t.expectEqual(want_words, layer.mixed.items.len);
    const used: u32 = BW.bitIdInWord(layer.bits_count);
    if (layer.bits_count > 0 and used != 0 and want_words > 0) {
        const valid = BW.maskStartClamped(used);
        try t.expectEqual(@as(Word, 0), layer.activity.items[want_words - 1] & ~valid);
        try t.expectEqual(@as(Word, 0), layer.mixed.items[want_words - 1] & ~valid);
    }
    const scanned = scanCounters(Word, layer);
    try t.expectEqualSlices(u32, &scanned, &layer.state_counters);
    var total: u32 = 0;
    for (layer.state_counters) |c| total += c;
    try t.expectEqual(layer.bits_count, total);
}

test "BitSet resize" {
    var bs = BitSet(u64){};
    defer bs.deinit(t.allocator);

    try bs.resize(t.allocator, 100, BitState.active);
    try t.expectEqual(100, bs.bits_count);
    try t.expectEqual(100, bs.active_bits_counter);
    try t.expectEqual(2, bs.words.items.len);

    try bs.resize(t.allocator, 127, BitState.inactive);
    try t.expectEqual(127, bs.bits_count);
    try t.expectEqual(100, bs.active_bits_counter);
    try t.expectEqual(2, bs.words.items.len);

    try bs.resize(t.allocator, 80, BitState.inactive);
    try t.expectEqual(80, bs.bits_count);
    try t.expectEqual(80, bs.active_bits_counter);
    try t.expectEqual(2, bs.words.items.len);

    try bs.resize(t.allocator, 10, BitState.inactive);
    try t.expectEqual(10, bs.bits_count);
    try t.expectEqual(10, bs.active_bits_counter);
    try t.expectEqual(1, bs.words.items.len);

    try bs.resize(t.allocator, 60, BitState.inactive);
    try t.expectEqual(60, bs.bits_count);
    try t.expectEqual(10, bs.active_bits_counter);
    try t.expectEqual(1, bs.words.items.len);

    try bs.resize(t.allocator, 200, BitState.active);
    try t.expectEqual(200, bs.bits_count);
    try t.expectEqual(150, bs.active_bits_counter);
    try t.expectEqual(4, bs.words.items.len);

    try bs.resize(t.allocator, 10000, BitState.inactive);
    try t.expectEqual(10000, bs.bits_count);
    try t.expectEqual(150, bs.active_bits_counter);
    try t.expectEqual(157, bs.words.items.len);

    try bs.resize(t.allocator, 2000, BitState.active);
    try t.expectEqual(2000, bs.bits_count);
    try t.expectEqual(150, bs.active_bits_counter);
    try t.expectEqual(32, bs.words.items.len);

    try bs.resize(t.allocator, 3000, BitState.active);
    try t.expectEqual(3000, bs.bits_count);
    try t.expectEqual(1150, bs.active_bits_counter);
    try t.expectEqual(47, bs.words.items.len);
}

test "Layer.State int constants match enum values" {
    const S = Layer(u64).State;
    try t.expectEqual(S.inactive_u2, @intFromEnum(S.inactive));
    try t.expectEqual(S.active_u2, @intFromEnum(S.active));
    try t.expectEqual(S.mixed_u2, @intFromEnum(S.mixed));
    try t.expectEqual(S.deep_mixed_u2, @intFromEnum(S.deep_mixed));
    try t.expectEqual(S.inactive_u32, @intFromEnum(S.inactive));
    try t.expectEqual(S.active_u32, @intFromEnum(S.active));
    try t.expectEqual(S.mixed_u32, @intFromEnum(S.mixed));
    try t.expectEqual(S.deep_mixed_u32, @intFromEnum(S.deep_mixed));
    comptime {
        if (S.inactive_u2 != @intFromEnum(S.inactive)) @compileError("inactive const mismatch");
        if (S.active_u2 != @intFromEnum(S.active)) @compileError("active const mismatch");
        if (S.mixed_u2 != @intFromEnum(S.mixed)) @compileError("mixed const mismatch");
        if (S.deep_mixed_u2 != @intFromEnum(S.deep_mixed)) @compileError("deep_mixed const mismatch");
    }
}

test "Layer(u8) countStates" {
    const L8 = Layer(u8);
    // a=0b1010, m=0b1100, full mask: b0=inactive, b1=active, b2=mixed,
    // b3=deep, upper zero bits b4..b7=inactive.
    const c = L8.countStates(0b1010, 0b1100, 0xFF);
    try t.expectEqual(@as(u32, 5), c.inactive);
    try t.expectEqual(@as(u32, 1), c.active);
    try t.expectEqual(@as(u32, 1), c.mixed);
    try t.expectEqual(@as(u32, 1), c.deep);

    // Restricted mask: only low 2 bits (b0=inactive, b1=active).
    const c2 = L8.countStates(0b1010, 0b1100, 0x03);
    try t.expectEqual(@as(u32, 1), c2.inactive);
    try t.expectEqual(@as(u32, 1), c2.active);
    try t.expectEqual(@as(u32, 0), c2.mixed);
    try t.expectEqual(@as(u32, 0), c2.deep);
}

test "Layer(u64) resize grow: allocation + counters" {
    var layer: Layer(u64) = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, 10, .inactive);
    try t.expectEqual(@as(u32, 10), layer.bits_count);
    try t.expectEqual([4]u32{ 10, 0, 0, 0 }, layer.state_counters);
    try expectLayerValid(u64, &layer);

    try layer.resize(t.allocator, 70, .active);
    try t.expectEqual(@as(u32, 70), layer.bits_count);
    try t.expectEqual([4]u32{ 10, 60, 0, 0 }, layer.state_counters);
    try expectLayerValid(u64, &layer);

    try layer.resize(t.allocator, 130, .mixed);
    try t.expectEqual([4]u32{ 10, 60, 60, 0 }, layer.state_counters);
    try expectLayerValid(u64, &layer);

    try layer.resize(t.allocator, 200, .deep_mixed);
    try t.expectEqual([4]u32{ 10, 60, 60, 70 }, layer.state_counters);
    try expectLayerValid(u64, &layer);
}

test "Layer(u64) resize grow within one word" {
    var layer: Layer(u64) = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, 5, .active);
    try layer.resize(t.allocator, 9, .mixed);
    try t.expectEqual([4]u32{ 0, 5, 4, 0 }, layer.state_counters);
    try expectLayerValid(u64, &layer);
}

test "Layer(u64) resize shrink: counters follow removed bits" {
    var layer: Layer(u64) = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, 200, .inactive);
    for (0..10) |i| {
        layer.activity.items[0] |= @as(u64, 1) << @intCast(i);
        layer.state_counters[Layer(u64).State.inactive_u32] -= 1;
        layer.state_counters[Layer(u64).State.active_u32] += 1;
    }
    for (10..20) |i| {
        layer.mixed.items[0] |= @as(u64, 1) << @intCast(i);
        layer.state_counters[Layer(u64).State.inactive_u32] -= 1;
        layer.state_counters[Layer(u64).State.mixed_u32] += 1;
    }
    for (20..30) |i| {
        layer.activity.items[0] |= @as(u64, 1) << @intCast(i);
        layer.mixed.items[0] |= @as(u64, 1) << @intCast(i);
        layer.state_counters[Layer(u64).State.inactive_u32] -= 1;
        layer.state_counters[Layer(u64).State.deep_mixed_u32] += 1;
    }
    try expectLayerValid(u64, &layer);

    try layer.resize(t.allocator, 25, .inactive);
    try t.expectEqual(@as(u32, 25), layer.bits_count);
    try t.expectEqual([4]u32{ 0, 10, 10, 5 }, layer.state_counters);
    try expectLayerValid(u64, &layer);

    try layer.resize(t.allocator, 10, .inactive);
    try t.expectEqual([4]u32{ 0, 10, 0, 0 }, layer.state_counters);
    try expectLayerValid(u64, &layer);

    try layer.resize(t.allocator, 0, .inactive);
    try t.expectEqual([4]u32{ 0, 0, 0, 0 }, layer.state_counters);
    try t.expectEqual(@as(usize, 0), layer.activity.items.len);
    try t.expectEqual(@as(usize, 0), layer.mixed.items.len);
}

test "Layer(u8) resize small word: grow/shrink roundtrip" {
    var layer: Layer(u8) = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, 3, .active);
    try layer.resize(t.allocator, 20, .mixed);
    try t.expectEqual([4]u32{ 0, 3, 17, 0 }, layer.state_counters);
    try expectLayerValid(u8, &layer);
    try layer.resize(t.allocator, 8, .inactive);
    try t.expectEqual([4]u32{ 0, 3, 5, 0 }, layer.state_counters);
    try expectLayerValid(u8, &layer);
    try layer.resize(t.allocator, 8, .active);
    try t.expectEqual([4]u32{ 0, 3, 5, 0 }, layer.state_counters);
}

test "Layer(u64) setWord: masked insert + counters" {
    const L64 = Layer(u64);
    var layer: L64 = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, 128, .inactive);
    try t.expectEqual([4]u32{ 128, 0, 0, 0 }, layer.state_counters);

    // Full-word insert: low 64 bits -> 32 active + 32 mixed.
    var act: u64 = 0;
    var mix: u64 = 0;
    for (0..32) |i| act |= @as(u64, 1) << @intCast(i);
    for (32..64) |i| mix |= @as(u64, 1) << @intCast(i);
    layer.setWord(0, act, mix, std.math.maxInt(u64));
    try t.expectEqual([4]u32{ 64, 32, 32, 0 }, layer.state_counters);
    try expectLayerValid(u64, &layer);

    // Partial mask: flip bits [0,4) to deep_mixed.
    layer.setWord(0, std.math.maxInt(u64), std.math.maxInt(u64), 0xF);
    try t.expectEqual([4]u32{ 64, 28, 32, 4 }, layer.state_counters);
    try expectLayerValid(u64, &layer);

    // Empty mask is a no-op.
    layer.setWord(0, 0, 0, 0);
    try t.expectEqual([4]u32{ 64, 28, 32, 4 }, layer.state_counters);

    // Same-value insert is a no-op for counters.
    layer.setWord(0, act | 0xF, mix | 0xF, std.math.maxInt(u64));
    try t.expectEqual([4]u32{ 64, 28, 32, 4 }, layer.state_counters);
    try expectLayerValid(u64, &layer);

    // Second word: single-bit insert via mask.
    layer.setWord(1, std.math.maxInt(u64), 0, 0x1);
    try t.expectEqual([4]u32{ 63, 29, 32, 4 }, layer.state_counters);
    try expectLayerValid(u64, &layer);
}

test "Layer(u64) setWord: padding bits ignored" {
    const L64 = Layer(u64);
    var layer: L64 = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, 70, .inactive); // last word has 6 valid bits
    layer.setWord(1, std.math.maxInt(u64), std.math.maxInt(u64), std.math.maxInt(u64));
    // Only 6 valid bits counted as deep, padding stays zero.
    try t.expectEqual([4]u32{ 64, 0, 0, 6 }, layer.state_counters);
    try expectLayerValid(u64, &layer);
    try t.expectEqual(@as(u64, 0x3F), layer.activity.items[1]);
    try t.expectEqual(@as(u64, 0x3F), layer.mixed.items[1]);
}

// ================= BitSet resize: exhaustive oracle corner-case tests =================
// NOTE: resize() intentionally NOT touched. Only tests below are new.
// Oracle idea: valid bits [0, bits_count) must satisfy:
//   * words.len == ceil(bits_count / WORD_BITS)  (0 -> 0)
//   * padding bits [bits_count%W .. W) in last word == 0
//   * active_bits_counter == popcount(valid bits)
//   * resize preserves prefix [0, min(old,new)), fills (old,new] with created, drops tail.
// Corner matrix covered:
//   grow vs shrink vs no-op | empty (0) vs non-empty | old aligned vs unaligned |
//   new aligned vs unaligned | same word-count vs cross-word | by 1 at boundary |
//   to zero / from zero | single word vs multi-word | all-ones/all-zeros/alt/pseudo
//   contents (removed range with 1s exposes counter bugs, padding exposes mask bugs) |
//   Word = u8/u16/u32/u64 (different shifts) | sequential chains (stale garbage).

fn bitSetScanActiveCount(comptime Word: type, bs: *const BitSet(Word)) u32 {
    const BW = bit_word.BitWord(Word);
    var acc: u32 = 0;
    var i: u32 = 0;
    while (i < bs.bits_count) : (i += 1) {
        const wid: usize = @intCast(BW.bitToWordId(i));
        const bid = BW.bitIdInWord(i);
        if (BW.readBit(bs.words.items[wid], bid) == .active) acc += 1;
    }
    return acc;
}

fn expectBitSetInvariants(comptime Word: type, bs: *const BitSet(Word)) !void {
    const BW = bit_word.BitWord(Word);
    const want_words: usize = @intCast(BW.bitsToWordsCount(bs.bits_count));
    try t.expectEqual(want_words, bs.words.items.len);
    if (bs.bits_count == 0) {
        try t.expectEqual(@as(u32, 0), bs.active_bits_counter);
        return;
    }
    const used: u32 = @as(u32, BW.bitIdInWord(bs.bits_count));
    if (used != 0) {
        const valid = BW.maskStartClamped(used);
        const last = bs.words.items[want_words - 1];
        try t.expectEqual(@as(Word, 0), last & ~valid);
    }
    const scanned = bitSetScanActiveCount(Word, bs);
    try t.expectEqual(scanned, bs.active_bits_counter);
    try t.expect(bs.active_bits_counter <= bs.bits_count);
}

const ResizePattern = enum { zero, one, alt01, alt10, every3, pseudo };

fn patternBit(p: ResizePattern, i: u32) BitState {
    return switch (p) {
        .zero => .inactive,
        .one => .active,
        .alt01 => if ((i & 1) == 0) .active else .inactive,
        .alt10 => if ((i & 1) == 0) .inactive else .active,
        .every3 => if ((i % 3) == 0) .active else .inactive,
        .pseudo => if ((((i *% 1664525) +% 1013904223) >> 15) & 1 == 1) .active else .inactive,
    };
}

fn initBitSetPattern(comptime Word: type, allocator: Allocator, bits: u32, pat: ResizePattern) !BitSet(Word) {
    var bs = BitSet(Word){};
    // Setup via inactive grow: padding-safe even with buggy resize (fill 0).
    try bs.resize(allocator, bits, .inactive);
    const BW = bit_word.BitWord(Word);
    var want_active: u32 = 0;
    var i: u32 = 0;
    while (i < bits) : (i += 1) {
        if (patternBit(pat, i) == .active) {
            const wid: usize = @intCast(BW.bitToWordId(i));
            const bid = BW.bitIdInWord(i);
            bs.words.items[wid] |= (@as(Word, 1) << bid);
            want_active += 1;
        }
    }
    bs.active_bits_counter = want_active;
    try expectBitSetInvariants(Word, &bs);
    return bs;
}

fn checkOneResize(comptime Word: type, allocator: Allocator, old: u32, new: u32, created: BitState, pat: ResizePattern) !void {
    var bs = try initBitSetPattern(Word, allocator, old, pat);
    defer bs.deinit(allocator);

    var expected_active: u32 = 0;
    const min_len = @min(old, new);
    var k: u32 = 0;
    while (k < min_len) : (k += 1) {
        if (patternBit(pat, k) == .active) expected_active += 1;
    }
    if (new > old and created == .active) expected_active += new - old;

    errdefer std.debug.print(
        "CTX Word={s} old={} new={} created={s} pat={s} bits={} counter={} expected_active={}\n",
        .{ @typeName(Word), old, new, @tagName(created), @tagName(pat), bs.bits_count, bs.active_bits_counter, expected_active },
    );

    try bs.resize(allocator, new, created);

    const BW = bit_word.BitWord(Word);
    try t.expectEqual(new, bs.bits_count);
    const want_words: usize = @intCast(BW.bitsToWordsCount(new));
    try t.expectEqual(want_words, bs.words.items.len);
    try t.expectEqual(expected_active, bs.active_bits_counter);

    var j: u32 = 0;
    while (j < new) : (j += 1) {
        const want: BitState = if (j < old) patternBit(pat, j) else created;
        const wid: usize = @intCast(BW.bitToWordId(j));
        const bid = BW.bitIdInWord(j);
        const got = BW.readBit(bs.words.items[wid], bid);
        try t.expectEqual(want, got);
    }
    try expectBitSetInvariants(Word, &bs);
}

fn nextRandU32(state: *u64) u32 {
    var x = state.*;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state.* = x;
    return @truncate((x *% 0x2545F4914F6CDD1D) >> 32);
}

test "BitSet resize minimal: grow active unaligned must keep padding zero" {
    // 0 -> 10 active (u64): valid [0,10)=1, padding [10,64) must be 0.
    var bs = BitSet(u64){};
    defer bs.deinit(t.allocator);
    try bs.resize(t.allocator, 10, .active);
    try t.expectEqual(@as(u32, 10), bs.bits_count);
    try t.expectEqual(@as(u32, 10), bs.active_bits_counter);
    try t.expectEqual(@as(usize, 1), bs.words.items.len);
    const BW = bit_word.BitWord(u64);
    try t.expectEqual(@as(u64, 0), bs.words.items[0] & ~BW.maskStartClamped(10));
    try expectBitSetInvariants(u64, &bs);

    // 0 -> 65 active: second word has 1 valid bit, rest padding.
    var bs2 = BitSet(u64){};
    defer bs2.deinit(t.allocator);
    try bs2.resize(t.allocator, 65, .active);
    try t.expectEqual(@as(u32, 65), bs2.active_bits_counter);
    try t.expectEqual(@as(u64, 0), bs2.words.items[1] & ~BW.maskStartClamped(1));
    try expectBitSetInvariants(u64, &bs2);
}

test "BitSet resize minimal: shrink to aligned must fix counter" {
    // 128 ones -> 64 : removed [64,128) = 64 ones.
    {
        var bs = BitSet(u64){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 128, .active);
        try t.expectEqual(@as(u32, 128), bs.active_bits_counter);
        try bs.resize(t.allocator, 64, .inactive);
        try t.expectEqual(@as(u32, 64), bs.bits_count);
        try t.expectEqual(@as(usize, 1), bs.words.items.len);
        try t.expectEqual(@as(u32, 64), bs.active_bits_counter);
        try expectBitSetInvariants(u64, &bs);
    }
    // 64 ones -> 0 : everything removed.
    {
        var bs = BitSet(u64){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 64, .active);
        try bs.resize(t.allocator, 0, .inactive);
        try t.expectEqual(@as(u32, 0), bs.bits_count);
        try t.expectEqual(@as(usize, 0), bs.words.items.len);
        try t.expectEqual(@as(u32, 0), bs.active_bits_counter);
    }
    // 65 -> 64 : shrink by 1 at word boundary.
    {
        var bs = try initBitSetPattern(u64, t.allocator, 65, .one);
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 64, .inactive);
        try t.expectEqual(@as(u32, 64), bs.active_bits_counter);
        try expectBitSetInvariants(u64, &bs);
    }
    // u8 version: 9 -> 8 (cross-word to aligned).
    {
        var bs = try initBitSetPattern(u8, t.allocator, 9, .one);
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 8, .inactive);
        try t.expectEqual(@as(u32, 8), bs.active_bits_counter);
        try expectBitSetInvariants(u8, &bs);
    }
}

test "BitSet resize minimal: shrink same-word must clear tail" {
    // 20 ones -> 10 (same u64 word): [10,20) becomes padding, must be 0.
    var bs = try initBitSetPattern(u64, t.allocator, 20, .one);
    defer bs.deinit(t.allocator);
    try bs.resize(t.allocator, 10, .inactive);
    try t.expectEqual(@as(u32, 10), bs.active_bits_counter);
    try expectBitSetInvariants(u64, &bs);
}

test "BitSet(u8) resize exhaustive 0..20 oracle" {
    const patterns = [_]ResizePattern{ .zero, .one, .alt01, .alt10, .every3, .pseudo };
    const createds = [_]BitState{ .inactive, .active };
    for (0..21) |old_usize| {
        for (0..21) |new_usize| {
            for (createds) |created| {
                for (patterns) |pat| {
                    try checkOneResize(u8, t.allocator, @intCast(old_usize), @intCast(new_usize), created, pat);
                }
            }
        }
    }
}

test "BitSet(u64) resize boundary oracle" {
    const bounds = [_]u32{ 0, 1, 2, 63, 64, 65, 70, 100, 126, 127, 128, 129, 130, 191, 192, 193, 200 };
    const patterns = [_]ResizePattern{ .zero, .one, .alt01, .alt10, .every3, .pseudo };
    const createds = [_]BitState{ .inactive, .active };
    for (bounds) |old| {
        for (bounds) |new| {
            for (createds) |created| {
                for (patterns) |pat| {
                    try checkOneResize(u64, t.allocator, old, new, created, pat);
                }
            }
        }
    }
}

test "BitSet(u32) resize boundary oracle" {
    const bounds = [_]u32{ 0, 1, 2, 31, 32, 33, 40, 63, 64, 65, 95, 96, 97, 100 };
    const patterns = [_]ResizePattern{ .zero, .one, .alt01, .alt10, .every3, .pseudo };
    const createds = [_]BitState{ .inactive, .active };
    for (bounds) |old| {
        for (bounds) |new| {
            for (createds) |created| {
                for (patterns) |pat| {
                    try checkOneResize(u32, t.allocator, old, new, created, pat);
                }
            }
        }
    }
}

test "BitSet(u16) resize boundary oracle" {
    const bounds = [_]u32{ 0, 1, 2, 15, 16, 17, 24, 31, 32, 33, 47, 48, 49 };
    const patterns = [_]ResizePattern{ .zero, .one, .alt01, .alt10, .every3, .pseudo };
    const createds = [_]BitState{ .inactive, .active };
    for (bounds) |old| {
        for (bounds) |new| {
            for (createds) |created| {
                for (patterns) |pat| {
                    try checkOneResize(u16, t.allocator, old, new, created, pat);
                }
            }
        }
    }
}

test "BitSet resize explicit corner table u64" {
    // {old, new, created, pattern}: covers empty, aligned/unaligned, same/cross, by-1, to-zero.
    const Case = struct { old: u32, new: u32, created: BitState, pat: ResizePattern };
    const cases = [_]Case{
        .{ .old = 0, .new = 0, .created = .active, .pat = .zero }, // no-op empty
        .{ .old = 0, .new = 1, .created = .active, .pat = .zero }, // from empty, 1 bit
        .{ .old = 0, .new = 1, .created = .inactive, .pat = .zero },
        .{ .old = 0, .new = 63, .created = .active, .pat = .zero }, // from empty to W-1
        .{ .old = 0, .new = 64, .created = .active, .pat = .zero }, // from empty to aligned
        .{ .old = 0, .new = 65, .created = .active, .pat = .zero }, // from empty to W+1
        .{ .old = 0, .new = 128, .created = .active, .pat = .zero }, // from empty multi-word aligned
        .{ .old = 1, .new = 0, .created = .inactive, .pat = .one }, // to zero
        .{ .old = 64, .new = 0, .created = .inactive, .pat = .one }, // aligned to zero
        .{ .old = 65, .new = 0, .created = .inactive, .pat = .one }, // unaligned to zero
        .{ .old = 10, .new = 10, .created = .active, .pat = .alt01 }, // no-op (created ignored)
        .{ .old = 64, .new = 64, .created = .active, .pat = .one }, // no-op aligned
        .{ .old = 10, .new = 20, .created = .active, .pat = .zero }, // grow same word
        .{ .old = 20, .new = 10, .created = .inactive, .pat = .one }, // shrink same word
        .{ .old = 10, .new = 64, .created = .inactive, .pat = .one }, // grow unaligned -> aligned same words
        .{ .old = 64, .new = 10, .created = .inactive, .pat = .one }, // shrink aligned -> unaligned same words? (2 words? 64=1w,10=1w same)
        .{ .old = 60, .new = 65, .created = .active, .pat = .alt01 }, // grow cross-word by 5
        .{ .old = 65, .new = 60, .created = .inactive, .pat = .one }, // shrink cross-word
        .{ .old = 10, .new = 70, .created = .active, .pat = .zero }, // grow 1w -> 2w unaligned
        .{ .old = 70, .new = 10, .created = .inactive, .pat = .one }, // shrink 2w -> 1w
        .{ .old = 64, .new = 65, .created = .active, .pat = .one }, // grow aligned by 1 (new word)
        .{ .old = 65, .new = 64, .created = .inactive, .pat = .one }, // shrink by 1 to aligned
        .{ .old = 63, .new = 64, .created = .active, .pat = .alt10 }, // grow to aligned same word-count
        .{ .old = 64, .new = 63, .created = .inactive, .pat = .one }, // shrink aligned by 1
        .{ .old = 127, .new = 128, .created = .active, .pat = .one }, // grow to aligned cross
        .{ .old = 128, .new = 127, .created = .inactive, .pat = .one }, // shrink from aligned by 1
        .{ .old = 128, .new = 129, .created = .active, .pat = .pseudo }, // grow from aligned
        .{ .old = 129, .new = 128, .created = .inactive, .pat = .pseudo }, // shrink to aligned
        .{ .old = 64, .new = 128, .created = .active, .pat = .alt01 }, // aligned -> aligned multi
        .{ .old = 128, .new = 64, .created = .inactive, .pat = .one }, // aligned -> aligned shrink (loop bug)
        .{ .old = 192, .new = 64, .created = .inactive, .pat = .one }, // multi-word drop (loop misses last)
        .{ .old = 200, .new = 64, .created = .inactive, .pat = .one }, // unaligned old -> aligned new
        .{ .old = 70, .new = 64, .created = .inactive, .pat = .one }, // unaligned -> aligned same word-count? (2w->1w)
        .{ .old = 100, .new = 70, .created = .inactive, .pat = .one }, // unaligned -> unaligned cross
        .{ .old = 2000, .new = 3000, .created = .active, .pat = .pseudo }, // large grow
        .{ .old = 3000, .new = 2000, .created = .inactive, .pat = .one }, // large shrink
        .{ .old = 10000, .new = 2000, .created = .inactive, .pat = .pseudo }, // large shrink to unaligned
    };
    for (cases) |c| {
        try checkOneResize(u64, t.allocator, c.old, c.new, c.created, c.pat);
    }
}

test "BitSet(u64) resize sequential fuzz vs oracle" {
    var bs = BitSet(u64){};
    defer bs.deinit(t.allocator);
    var expected: [512]u1 = [_]u1{0} ** 512;
    var expected_len: u32 = 0;
    var expected_active: u32 = 0;
    var rng: u64 = 0x9E3779B97F4A7C15;
    const BW = bit_word.BitWord(u64);
    var step: usize = 0;
    while (step < 500) : (step += 1) {
        const r1 = nextRandU32(&rng);
        const r2 = nextRandU32(&rng);
        const new_len: u32 = r1 % 201; // 0..200 : zero, aligned(64/128/192), unaligned, same/cross
        const created: BitState = if ((r2 & 1) == 1) .active else .inactive;
        if (new_len > expected_len) {
            for (expected_len..new_len) |i| {
                expected[i] = @intFromEnum(created);
                if (created == .active) expected_active += 1;
            }
        } else if (new_len < expected_len) {
            for (new_len..expected_len) |i| {
                if (expected[i] == 1) expected_active -= 1;
            }
        }
        expected_len = new_len;
        errdefer std.debug.print("SEQ u64 fail step={} new={} created={s}\n", .{ step, new_len, @tagName(created) });
        try bs.resize(t.allocator, new_len, created);
        try t.expectEqual(expected_len, bs.bits_count);
        try t.expectEqual(@as(usize, @intCast(BW.bitsToWordsCount(expected_len))), bs.words.items.len);
        try t.expectEqual(expected_active, bs.active_bits_counter);
        for (0..expected_len) |idx| {
            const i: u32 = @intCast(idx);
            const want: BitState = @enumFromInt(expected[idx]);
            const wid: usize = @intCast(BW.bitToWordId(i));
            const got = BW.readBit(bs.words.items[wid], BW.bitIdInWord(i));
            try t.expectEqual(want, got);
        }
        try expectBitSetInvariants(u64, &bs);
    }
}

test "BitSet(u8) resize sequential fuzz vs oracle" {
    var bs = BitSet(u8){};
    defer bs.deinit(t.allocator);
    var expected: [64]u1 = [_]u1{0} ** 64;
    var expected_len: u32 = 0;
    var expected_active: u32 = 0;
    var rng: u64 = 0x123456789ABCDEF;
    const BW = bit_word.BitWord(u8);
    var step: usize = 0;
    while (step < 300) : (step += 1) {
        const r1 = nextRandU32(&rng);
        const r2 = nextRandU32(&rng);
        const new_len: u32 = r1 % 41; // 0..40 : covers 0..5 words for u8
        const created: BitState = if ((r2 & 1) == 1) .active else .inactive;
        if (new_len > expected_len) {
            for (expected_len..new_len) |i| {
                expected[i] = @intFromEnum(created);
                if (created == .active) expected_active += 1;
            }
        } else if (new_len < expected_len) {
            for (new_len..expected_len) |i| {
                if (expected[i] == 1) expected_active -= 1;
            }
        }
        expected_len = new_len;
        errdefer std.debug.print("SEQ u8 fail step={} new={} created={s}\n", .{ step, new_len, @tagName(created) });
        try bs.resize(t.allocator, new_len, created);
        try t.expectEqual(expected_len, bs.bits_count);
        try t.expectEqual(@as(usize, @intCast(BW.bitsToWordsCount(expected_len))), bs.words.items.len);
        try t.expectEqual(expected_active, bs.active_bits_counter);
        for (0..expected_len) |idx| {
            const i: u32 = @intCast(idx);
            const want: BitState = @enumFromInt(expected[idx]);
            const wid: usize = @intCast(BW.bitToWordId(i));
            const got = BW.readBit(bs.words.items[wid], BW.bitIdInWord(i));
            try t.expectEqual(want, got);
        }
        try expectBitSetInvariants(u8, &bs);
    }
}

test "BitSet resize same-size no-op keeps storage" {
    const sizes = [_]u32{ 0, 1, 7, 8, 9, 64, 65, 128 };
    for (sizes) |n| {
        var bs = try initBitSetPattern(u64, t.allocator, n, .pseudo);
        defer bs.deinit(t.allocator);
        const old_counter = bs.active_bits_counter;
        const old_len = bs.words.items.len;
        // created value must be ignored when size unchanged
        try bs.resize(t.allocator, n, .active);
        try t.expectEqual(n, bs.bits_count);
        try t.expectEqual(old_counter, bs.active_bits_counter);
        try t.expectEqual(old_len, bs.words.items.len);
        try bs.resize(t.allocator, n, .inactive);
        try t.expectEqual(old_counter, bs.active_bits_counter);
        try expectBitSetInvariants(u64, &bs);
    }
}

test "BitSet resize large sizes oracle" {
    try checkOneResize(u64, t.allocator, 0, 10000, .active, .zero);
    try checkOneResize(u64, t.allocator, 0, 10000, .inactive, .zero);
    try checkOneResize(u64, t.allocator, 10000, 2000, .inactive, .one);
    try checkOneResize(u64, t.allocator, 2000, 3000, .active, .pseudo);
    try checkOneResize(u64, t.allocator, 3000, 64, .inactive, .one);
    try checkOneResize(u64, t.allocator, 64, 10000, .active, .alt01);
}
