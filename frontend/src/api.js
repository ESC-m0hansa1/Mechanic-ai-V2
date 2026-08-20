// One place that knows how to talk to the API. Components never call fetch
// directly, so error handling and the request-id header live in a single file.

/** Ask the manual a question. Throws Error(message) on any failure. */
export async function ask(question, k = 5) {
  let res;
  try {
    res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k }),
    });
  } catch {
    // fetch only rejects on network-level failure, never on 4xx/5xx.
    throw new Error("Cannot reach the server. Is the API running?");
  }

  if (!res.ok) {
    // FastAPI puts a safe, generic string in `detail` (502/503) or a validation
    // array (422). Never surface the raw body: it is not written for users.
    const body = await res.json().catch(() => ({}));
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : res.status === 422
          ? "That question was rejected: it must be 3-500 characters."
          : `Request failed (${res.status}).`,
    );
  }
  return res.json();
}

/** Readiness plus the active configuration, for the status strip. */
export async function health() {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error("health check failed");
  return res.json();
}
