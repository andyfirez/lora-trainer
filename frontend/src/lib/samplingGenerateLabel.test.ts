import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { resolveGenerateBusy, resolveGenerateLabel } from "./samplingGenerateLabel.ts";

describe("resolveGenerateLabel", () => {
  it("is Generate when idle", () => {
    assert.equal(
      resolveGenerateLabel({
        optimistic: false,
        sessionQueued: false,
        sessionRunning: false,
      }),
      "Generate",
    );
  });

  it("is Interrupt immediately when optimistic", () => {
    assert.equal(
      resolveGenerateLabel({
        optimistic: true,
        sessionQueued: false,
        sessionRunning: false,
      }),
      "Interrupt",
    );
  });

  it("is Interrupt when this session is queued", () => {
    assert.equal(
      resolveGenerateLabel({
        optimistic: false,
        sessionQueued: true,
        sessionRunning: false,
      }),
      "Interrupt",
    );
  });

  it("is Interrupt when this session is running", () => {
    assert.equal(
      resolveGenerateLabel({
        optimistic: false,
        sessionQueued: false,
        sessionRunning: true,
      }),
      "Interrupt",
    );
  });
});

describe("resolveGenerateBusy", () => {
  it("never disables Interrupt", () => {
    assert.equal(resolveGenerateBusy({ label: "Interrupt", submitting: true }), false);
    assert.equal(resolveGenerateBusy({ label: "Interrupt", submitting: false }), false);
  });

  it("disables Generate while submitting", () => {
    assert.equal(resolveGenerateBusy({ label: "Generate", submitting: true }), true);
    assert.equal(resolveGenerateBusy({ label: "Generate", submitting: false }), false);
  });
});
