const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("st_ecs", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    const bit_word_mod = b.addModule("bit_word", .{
        .root_source_file = b.path("src/bit_word.zig"),
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

    const bit_tree_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bit_tree.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_bit_tree_tests = b.addRunArtifact(bit_tree_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_bit_tree_tests.step);

    const bench_exe = b.addExecutable(.{
        .name = "bench_flat_bit_set",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench_flat_bit_set.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_bench = b.addRunArtifact(bench_exe);

    const bench_step = b.step("bench", "Run FlatBitSet benchmarks");
    bench_step.dependOn(&run_bench.step);

    const iter_bench_exe = b.addExecutable(.{
        .name = "bench_iter",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/iteration_tests/bench.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bit_word", .module = bit_word_mod },
            },
        }),
    });

    const run_iter_bench = b.addRunArtifact(iter_bench_exe);

    const iter_bench_step = b.step("bench-iter", "Run iteration strategy benchmarks");
    iter_bench_step.dependOn(&run_iter_bench.step);

    const bench8_exe = b.addExecutable(.{
        .name = "bench_p8",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/iteration_tests/bench8.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bit_word", .module = bit_word_mod },
            },
        }),
    });

    const run_bench8 = b.addRunArtifact(bench8_exe);

    const bench8_step = b.step("bench-p8", "Run p8 iterateByte benchmark only");
    bench8_step.dependOn(&run_bench8.step);

    const iter_test_step = b.step("test-iter", "Run iteration strategy tests");
    inline for (&.{ "base", "p3", "p8", "p9" }) |name| {
        const iter_tests = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("src/iteration_tests/{s}.zig", .{name})),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "bit_word", .module = bit_word_mod },
                },
            }),
        });
        iter_test_step.dependOn(&b.addRunArtifact(iter_tests).step);
    }

    // Asm-дампы стратегий для сравнения оптимизаций (zig-out/asm/*.s).
    const iter_asm_step = b.step("asm-iter", "Emit iteration strategy assembly");
    inline for (&.{ "base", "p3", "p8", "p9" }) |name| {
        const iter_obj = b.addObject(.{
            .name = b.fmt("iter_{s}", .{name}),
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("src/iteration_tests/{s}.zig", .{name})),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "bit_word", .module = bit_word_mod },
                },
            }),
        });
        const install_asm = b.addInstallFile(
            iter_obj.getEmittedAsm(),
            b.fmt("asm/{s}.s", .{name}),
        );
        iter_asm_step.dependOn(&install_asm.step);
    }
}
