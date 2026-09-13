const std = @import("std");
const utilities = @import("utilities.zig");

const ListA64 = utilities.ListA64;

pub fn ECSTable(comptime instance_id: u32) type {
    const result = struct {
        pub const table_instance_id = instance_id;

        var entity_generations: ListA64(u8) = .empty;
    };

    return result;
}
