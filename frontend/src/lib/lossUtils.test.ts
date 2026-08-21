import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { mergeLossSeries } from "./lossUtils.ts";

describe("mergeLossSeries", () => {
  it("replaces series on initial load", () => {
    const result = mergeLossSeries({
      previous: { "loss/loss": [{ step: 0, value: 9 }] },
      results: [{ key: "loss/loss", keys: ["loss/loss"], points: [{ step: 1, value: 0.5 }] }],
      wantedKeys: ["loss/loss"],
      isInitialLoad: true,
      lastStepByKey: {},
    });

    assert.deepEqual(result.next["loss/loss"], [{ step: 1, value: 0.5 }]);
    assert.equal(result.lastStepByKey["loss/loss"], 1);
  });

  it("appends only newer steps during refresh", () => {
    const result = mergeLossSeries({
      previous: { "loss/loss": [{ step: 1, value: 0.5 }] },
      results: [
        {
          key: "loss/loss",
          keys: ["loss/loss"],
          points: [
            { step: 1, value: 0.5 },
            { step: 2, value: 0.4 },
          ],
        },
      ],
      wantedKeys: ["loss/loss"],
      isInitialLoad: false,
      lastStepByKey: { "loss/loss": 1 },
    });

    assert.deepEqual(result.next["loss/loss"], [
      { step: 1, value: 0.5 },
      { step: 2, value: 0.4 },
    ]);
  });

  it("drops series for keys that disappeared", () => {
    const result = mergeLossSeries({
      previous: {
        "loss/loss": [{ step: 1, value: 0.5 }],
        "loss/avr_loss": [{ step: 1, value: 0.4 }],
      },
      results: [{ key: "loss/loss", keys: ["loss/loss"], points: [{ step: 2, value: 0.3 }] }],
      wantedKeys: ["loss/loss"],
      isInitialLoad: false,
      lastStepByKey: { "loss/loss": 1, "loss/avr_loss": 1 },
    });

    assert.equal(result.next["loss/avr_loss"], undefined);
    assert.equal(result.lastStepByKey["loss/avr_loss"], undefined);
  });
});
