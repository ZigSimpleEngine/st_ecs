const utilities = @import("utilities.zig");

pub fn Component(comptime table_instance_id_: u32) type {
    const result = struct {
        pub const table_instance_id = table_instance_id_;
    };

    return result;
}
