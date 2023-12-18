const std = @import("std");

const Allocator = std.mem.Allocator;
const print = std.debug.print;

// example translated to zig from https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

const AlphaBeta = struct {
    predictions: std.ArrayList(f32),
    estimates: std.ArrayList(f32),
    inputs: std.ArrayList(f32),
    gain_rate: f32,
    time_step: f32,
    scale_factor: f32,
    const Self = @This();
    pub fn init(allocator: std.mem.Allocator, initial_estimate: f32, gain_rate: f32, time_step: f32, scale_factor: f32) !Self {
        var self: Self = undefined;
        self.predictions = std.ArrayList(f32).init(allocator);
        self.estimates = std.ArrayList(f32).init(allocator);
        try self.estimates.append(initial_estimate);
        self.gain_rate = gain_rate;
        self.time_step = time_step;
        self.scale_factor = scale_factor;
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.predictions.deinit();
        self.estimates.deinit();
        //self.inputs.deinit();
    }

    fn save_plot_predictions(self: *Self, filename: []const u8) !void {
        var file = (try std.fs.cwd().createFile(filename, .{
            .truncate = true,
        })).writer();
        for (self.predictions.items, 1..) |prediction, i| {
            try file.print("{d:.2} {d:.2}\n", .{ i, prediction });
        }
    }

    fn save_plot_estimations(self: *Self, filename: []const u8) !void {
        var file = (try std.fs.cwd().createFile(filename, .{
            .truncate = true,
        })).writer();
        for (self.estimates.items, 0..) |estimation, i| {
            try file.print("{d:.2} {d:.2}\n", .{ i, estimation });
        }
    }

    pub fn predict_via_gain_guess(self: *Self, sensor_input: f32) !void {
        var predicted_weight: f32 = 0;
        var last_estimated_weight: f32 = self.estimates.getLast();
        predicted_weight = last_estimated_weight + self.gain_rate * self.time_step;
        var estimated_weight = predicted_weight + self.scale_factor * (sensor_input - predicted_weight);
        try self.estimates.append(estimated_weight);
        try self.predictions.append(predicted_weight);
        //print("\nPrevious estimate: {d:.2}, prediction: {d:.2}, estimate: {d:.2}", .{ last_estimated_weight, predicted_weight, estimated_weight });
    }
};

test "base alpha beta filter" {
    const time_step = 1;
    const scale_factor = 4.0 / 10.0;
    const sampleData = [_]f32{ 158.0, 164.2, 160.3, 159.9, 162.1, 164.6, 169.6, 167.4, 166.4, 171.0, 171.2, 172.6 };
    var initial_estimate: f32 = 160.0;
    var gain_rate: f32 = 1;
    var ab = try AlphaBeta.init(std.testing.allocator, initial_estimate, gain_rate, time_step, scale_factor);
    for (sampleData) |sample| {
        try ab.predict_via_gain_guess(sample);
    }
    print("len of estimates: {d}\n", .{ab.estimates.items.len});
    print("len of predictions: {d}\n", .{ab.predictions.items.len});
    ab.deinit();
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const time_step = 1;
    const scale_factor = 4.0 / 10.0;
    const sampleData = [_]f32{ 158.0, 164.2, 160.3, 159.9, 162.1, 164.6, 169.6, 167.4, 166.4, 171.0, 171.2, 172.6 };
    var initial_estimate: f32 = 160.0;
    var gain_rate: f32 = 1;
    var ab = try AlphaBeta.init(allocator, initial_estimate, gain_rate, time_step, scale_factor);
    for (sampleData) |sample| {
        try ab.predict_via_gain_guess(sample);
    }
    print("len of estimates: {d}\n", .{ab.estimates.items.len});
    print("len of predictions: {d}\n", .{ab.predictions.items.len});
    try ab.save_plot_predictions("predictions.dat");
    try ab.save_plot_estimations("estimations.dat");

    var file = (try std.fs.cwd().createFile("measurements.dat", .{ .truncate = true })).writer();
    for (sampleData, 0..) |measurement, i| {
        try file.print("{d:.2} {d:.2}\n", .{ i, measurement });
    }

    file = (try std.fs.cwd().createFile("actual.dat", .{ .truncate = true })).writer();
    for (160..172, 0..) |y, x| {
        try file.print("{d:.2} {d:.2}\n", .{ x, y });
    }
    ab.deinit();
}
