import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  PLAYGROUND_SESSION_ID_CAP,
  activePlaygroundJobs,
  mergeSessionIds,
  normalizeSessionIds,
} from "./samplingSession.ts";

function job(
  id: number,
  kind: "experiment" | "playground",
  status: string,
): { id: number; kind: "experiment" | "playground"; status: string } {
  return { id, kind, status };
}

describe("activePlaygroundJobs", () => {
  it("returns empty for missing list", () => {
    assert.deepEqual(activePlaygroundJobs(undefined), []);
  });

  it("keeps queued and running playground jobs", () => {
    const jobs = [
      job(1, "playground", "running"),
      job(2, "playground", "queued"),
      job(3, "playground", "completed"),
      job(4, "experiment", "running"),
      job(5, "playground", "draft"),
    ];
    assert.deepEqual(
      activePlaygroundJobs(jobs).map((item) => item.id),
      [1, 2],
    );
  });
});

describe("normalizeSessionIds", () => {
  it("drops invalid and duplicate ids", () => {
    assert.deepEqual(normalizeSessionIds([1, 1, 0, -4, 2.8, "3", "x", Number.NaN]), [1, 2, 3]);
  });

  it("caps to the most recent ids", () => {
    const ids = Array.from({ length: PLAYGROUND_SESSION_ID_CAP + 5 }, (_, index) => index + 1);
    const normalized = normalizeSessionIds(ids);
    assert.equal(normalized.length, PLAYGROUND_SESSION_ID_CAP);
    assert.equal(normalized[0], 6);
    assert.equal(normalized.at(-1), PLAYGROUND_SESSION_ID_CAP + 5);
  });

  it("returns empty for non-arrays", () => {
    assert.deepEqual(normalizeSessionIds(null), []);
    assert.deepEqual(normalizeSessionIds("1,2"), []);
  });
});

describe("mergeSessionIds", () => {
  it("appends active playground ids without duplicates", () => {
    const merged = mergeSessionIds(
      [10, 11],
      [job(11, "playground", "running"), job(12, "playground", "queued"), job(13, "experiment", "running")],
    );
    assert.deepEqual(merged, [10, 11, 12]);
  });

  it("ignores completed playground jobs", () => {
    assert.deepEqual(mergeSessionIds([4], [job(5, "playground", "completed")]), [4]);
  });
});
