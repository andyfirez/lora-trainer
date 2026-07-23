import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { computeProgressTiming } from "./progressTiming.ts";

describe("computeProgressTiming", () => {
  it("computes average speed and ETA from elapsed time", () => {
    const timing = computeProgressTiming(10, 100, 500, 40);

    assert.equal(timing.avgSecondsPerIteration, 50);
    assert.equal(timing.secondsPerIteration, 40);
    assert.equal(timing.elapsedSeconds, 500);
    assert.equal(timing.etaSeconds, 4500);
  });

  it("returns null avg and ETA when step is below 1", () => {
    const timing = computeProgressTiming(0, 100, 120, 30);

    assert.equal(timing.avgSecondsPerIteration, null);
    assert.equal(timing.etaSeconds, null);
    assert.equal(timing.secondsPerIteration, null);
  });

  it("returns null instant speed when step is below 2", () => {
    const timing = computeProgressTiming(1, 100, 60, 30);

    assert.equal(timing.avgSecondsPerIteration, 60);
    assert.equal(timing.secondsPerIteration, null);
    assert.equal(timing.etaSeconds, 5940);
  });

  it("returns null ETA when average speed is unavailable", () => {
    const timing = computeProgressTiming(null, 100, 0, null);

    assert.equal(timing.avgSecondsPerIteration, null);
    assert.equal(timing.etaSeconds, null);
  });
});
