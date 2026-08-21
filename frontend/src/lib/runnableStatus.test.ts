import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { canCancel, canEnqueue, cancelActionLabel } from "./runnableStatus.ts";

describe("runnableStatus", () => {
  it("canEnqueue returns true for restartable statuses", () => {
    for (const status of ["draft", "failed", "cancelled", "orphan"] as const) {
      assert.equal(canEnqueue(status), true);
    }
    assert.equal(canEnqueue("running"), false);
    assert.equal(canEnqueue("completed"), false);
  });

  it("canCancel returns true for queued and running", () => {
    assert.equal(canCancel("queued"), true);
    assert.equal(canCancel("running"), true);
    assert.equal(canCancel("draft"), false);
  });

  it("cancelActionLabel distinguishes running from queued", () => {
    assert.equal(cancelActionLabel("running"), "Stop");
    assert.equal(cancelActionLabel("queued"), "Cancel");
  });
});
