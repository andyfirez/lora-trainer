import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { ApiError, parseApiError } from "./errors.ts";

describe("parseApiError", () => {
  it("formats FastAPI validation detail arrays", () => {
    const error = parseApiError(
      { detail: [{ msg: "Field required" }, { msg: "Invalid value" }] },
      422,
    );
    assert.equal(error.message, "Field required, Invalid value");
    assert.equal(error.status, 422);
    assert.ok(error instanceof ApiError);
  });

  it("falls back to string detail or HTTP status", () => {
    assert.equal(parseApiError({ detail: "Not found" }, 404).message, "Not found");
    assert.equal(parseApiError({}, 500).message, "HTTP 500");
  });
});
