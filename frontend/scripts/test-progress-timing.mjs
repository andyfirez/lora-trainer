/**
 * Unit tests for progressTiming helpers.
 * Run: node --experimental-strip-types scripts/test-progress-timing.mjs
 */

import assert from "node:assert/strict";
import {
  computeProgressTiming,
  formatDuration,
  formatIterationSeconds,
} from "../src/lib/progressTiming.ts";

assert.equal(formatDuration(null), "—");
assert.equal(formatIterationSeconds(null), "—");

const noSteps = computeProgressTiming(null, 100, 0, 5);
assert.equal(noSteps.avgSecondsPerIteration, null);
assert.equal(noSteps.etaSeconds, null);
assert.equal(noSteps.secondsPerIteration, null);

const firstStep = computeProgressTiming(1, 100, 60, 10);
assert.equal(firstStep.avgSecondsPerIteration, 60);
assert.equal(firstStep.secondsPerIteration, null);
assert.equal(firstStep.etaSeconds, 99 * 60);

const withInstant = computeProgressTiming(50, 100, 500, 8);
assert.equal(withInstant.avgSecondsPerIteration, 10);
assert.equal(withInstant.secondsPerIteration, 8);
assert.equal(withInstant.etaSeconds, 50 * 10);

const completed = computeProgressTiming(100, 100, 1000, 5);
assert.equal(completed.etaSeconds, null);
assert.equal(completed.avgSecondsPerIteration, 10);

console.log("progressTiming tests passed");
