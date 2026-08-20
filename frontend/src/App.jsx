import React, { useEffect, useRef, useState } from "react";

import { ask, health } from "./api.js";
import Answer from "./components/Answer.jsx";
import Sources from "./components/Sources.jsx";

// Taken from eval/golden_set.json so the demo shows off the cases the retrieval
// eval actually measures: one exact-term, one paraphrase, one specs lookup, and
// one the manual cannot answer (the refusal path is a feature, not a failure).
const EXAMPLES = [
  "How do I check the engine oil level?",
  "My truck won't turn over when I press the start button. Why?",
  "What is the engine coolant capacity?",
  "How much is a used HILUX worth in Delhi?",
];

export default function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);
  const [activePage, setActivePage] = useState(null);
  const inputRef = useRef(null);

  useEffect(() => {
    // Fire and forget: the status strip is nice to have, so a failure here must
    // not block asking questions.
    health().then(setStatus).catch(() => setStatus(null));
  }, []);

  async function submit(text) {
    const q = (text ?? question).trim();
    if (q.length < 3 || loading) return;

    setQuestion(q);
    setLoading(true);
    setError(null);
    setActivePage(null);
    // Clear the old answer: leaving a stale one on screen next to a spinner
    // makes it look like the new question has already been answered.
    setResult(null);
    try {
      setResult(await ask(q, 5));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function useExample(text) {
    setQuestion(text);
    submit(text);
    inputRef.current?.focus();
  }

  return (
    <div className="page">
      <header className="masthead">
        <div className="brand">
          <span className="lamp" aria-hidden="true" />
          <h1>Mechanic AI</h1>
        </div>
        <p className="tagline">
          Repair answers grounded in a real Toyota HILUX owner&rsquo;s manual.
          Every claim is cited to a page, and questions the manual cannot answer
          get refused rather than invented.
        </p>
        {status && (
          <dl className="status" aria-label="system status">
            <div>
              <dt>corpus</dt>
              <dd>{status.chunks_retrievable ?? "-"} chunks</dd>
            </div>
            <div>
              <dt>retrieval</dt>
              <dd>{status.retrieval_strategy}</dd>
            </div>
            <div>
              <dt>embeddings</dt>
              <dd>{(status.embedding_model || "").split("/").pop()}</dd>
            </div>
            {status.reranker_model && (
              <div>
                <dt>reranker</dt>
                <dd>{status.reranker_model.split("/").pop()}</dd>
              </div>
            )}
          </dl>
        )}
      </header>

      <form
        className="asker"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <textarea
          ref={inputRef}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            // Enter submits, Shift+Enter makes a newline - the convention for a
            // single-question box that happens to be multi-line.
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder="Describe the problem, or ask about a procedure..."
          rows={3}
          maxLength={500}
          spellCheck="true"
          aria-label="Your question"
        />
        <div className="asker-row">
          <span className="counter">{question.length}/500</span>
          <button type="submit" disabled={loading || question.trim().length < 3}>
            {loading ? "Searching the manual..." : "Ask"}
          </button>
        </div>
      </form>

      <div className="examples">
        <span>Try:</span>
        {EXAMPLES.map((ex) => (
          <button key={ex} type="button" onClick={() => useExample(ex)} disabled={loading}>
            {ex}
          </button>
        ))}
      </div>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {loading && (
        <div className="skeleton" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
      )}

      {result && (
        <main className="result">
          <Answer result={result} onCite={setActivePage} />
          <Sources
            sources={result.sources}
            strategy={result.strategy}
            activePage={activePage}
          />
        </main>
      )}

      <footer className="foot">
        <span>
          Hybrid retrieval (dense + BM25 fused with Reciprocal Rank Fusion),
          reranked by a cross-encoder.
        </span>
        <a href="/docs">API docs</a>
      </footer>
    </div>
  );
}
