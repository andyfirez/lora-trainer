import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  computeAverageStepDuration,
  computeMovingAverageWindow,
  computeProgressTiming,
} from "./progressTiming.ts";

describe("computeMovingAverageWindow", () => {
  it("returns ceil(15% of total) with minimum 1", () => {
    assert.equal(computeMovingAverageWindow(100), 15);
    assert.equal(computeMovingAverageWindow(33), 5);
    assert.equal(computeMovingAverageWindow(6), 1);
    assert.equal(computeMovingAverageWindow(1), 1);
  });

  it("returns Infinity when total is unknown", () => {
    assert.equal(computeMovingAverageWindow(null), Infinity);
    assert.equal(computeMovingAverageWindow(0), Infinity);
  });
});

describe("computeAverageStepDuration", () => {
  it("averages the last N step durations", () => {
    const steady = Array(20).fill(50);
    assert.equal(computeAverageStepDuration(steady, 15), 50);
    assert.equal(computeAverageStepDuration([40, 40, 40, 40, 40], 5), 40);
  });

  it("uses all durations when fewer than window size", () => {
    assert.equal(computeAverageStepDuration([30, 50], 15), 40);
  });

  it("returns null for empty durations", () => {
    assert.equal(computeAverageStepDuration([], 15), null);
  });
});

describe("computeProgressTiming", () => {
  it("computes moving average speed and ETA from recent step durations", () => {
    const durations = Array(10).fill(50);
    const timing = computeProgressTiming(10, 100, 500, 40, durations);

    assert.equal(timing.avgSecondsPerIteration, 50);
    assert.equal(timing.secondsPerIteration, 40);
    assert.equal(timing.elapsedSeconds, 500);
    assert.equal(timing.etaSeconds, 4500);
  });

  it("uses only the recent window for ETA", () => {
    const durations = [...Array(15).fill(60), ...Array(15).fill(40)];
    const timing = computeProgressTiming(30, 100, 1500, 40, durations);

    assert.equal(timing.avgSecondsPerIteration, 40);
    assert.equal(timing.etaSeconds, 2800);
  });

  it("returns null avg and ETA when step is below 1", () => {
    const timing = computeProgressTiming(0, 100, 120, 30, [30]);

    assert.equal(timing.avgSecondsPerIteration, null);
    assert.equal(timing.etaSeconds, null);
    assert.equal(timing.secondsPerIteration, null);
  });

  it("returns null avg when no step durations are available yet", () => {
    const timing = computeProgressTiming(1, 100, 60, 30, []);

    assert.equal(timing.avgSecondsPerIteration, null);
    assert.equal(timing.secondsPerIteration, null);
    assert.equal(timing.etaSeconds, null);
  });

  it("returns null instant speed when step is below 2", () => {
    const timing = computeProgressTiming(1, 100, 60, 30, [60]);

    assert.equal(timing.avgSecondsPerIteration, 60);
    assert.equal(timing.secondsPerIteration, null);
    assert.equal(timing.etaSeconds, 5940);
  });

  it("falls back to all durations when total is unknown", () => {
    const durations = [30, 50, 70];
    const timing = computeProgressTiming(3, null, 150, 70, durations);

    assert.equal(timing.avgSecondsPerIteration, 50);
    assert.equal(timing.etaSeconds, null);
  });

  it("returns null ETA when average speed is unavailable", () => {
    const timing = computeProgressTiming(null, 100, 0, null, []);

    assert.equal(timing.avgSecondsPerIteration, null);
    assert.equal(timing.etaSeconds, null);
  });
});
