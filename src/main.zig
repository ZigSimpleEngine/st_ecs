const std = @import("std");
const Io = std.Io;

const st_ecs = @import("st_ecs");

pub fn main(init: std.process.Init) !void {
    _ = init;
    std.debug.print("Hello world", .{});
}
