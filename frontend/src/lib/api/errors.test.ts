import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { ApiError, parseApiError, parseNetworkError } from "./errors.ts";

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

describe("parseNetworkError", () => {
  it("maps Failed to fetch to a readable unreachable-API message", () => {
    const error = parseNetworkError(new TypeError("Failed to fetch"));
    assert.ok(error instanceof ApiError);
    assert.equal(error.message, "Cannot reach the API. Is the backend running?");
    assert.equal(error.status, 0);
  });

  it("keeps other network error messages", () => {
    const error = parseNetworkError(new Error("aborted"));
    assert.equal(error.message, "aborted");
    assert.equal(error.status, 0);
  });
});
