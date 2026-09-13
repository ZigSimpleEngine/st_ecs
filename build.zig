const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("st_ecs", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    const exe = b.addExecutable(.{
        .name = "st_ecs",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "st_ecs", .module = mod },
            },
        }),
    });

    const exe_install_artifact = b.addInstallArtifact(exe, .{});

    const run_artifact = b.addRunArtifact(exe);
    run_artifact.step.dependOn(&exe_install_artifact.step);

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_artifact.step);
}
