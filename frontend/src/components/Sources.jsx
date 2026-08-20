import React from "react";

// What the score means depends on which strategy produced it, so the UI says so
// instead of implying the numbers are comparable across strategies.
const SCORE_HELP = {
  dense: "Cosine similarity (0-1). Higher is closer in embedding space.",
  hybrid: "Reciprocal Rank Fusion score (~0.01-0.03). A fused rank, not a similarity.",
  reranked: "Cross-encoder logit (roughly -11 to +11). Not a probability.",
};

export default function Sources({ sources, strategy, activePage }) {
  if (!sources.length) return null;

  return (
    <section className="sources">
      <h2>
        Retrieved excerpts <span className="count">{sources.length}</span>
      </h2>
      <p className="sources-note" title={SCORE_HELP[strategy]}>
        Ranked by <strong>{strategy}</strong>. Scores are on this strategy&rsquo;s own
        scale.
      </p>
      <ol>
        {sources.map((s) => (
          <li
            key={s.chunk_id}
            className={activePage === s.page ? "source is-active" : "source"}
          >
            <div className="source-head">
              <span className="page">page {s.page}</span>
              <span className="score" title={SCORE_HELP[strategy]}>
                {s.score.toFixed(3)}
              </span>
              <span className="chunk-id" title="Primary key in the chunks table">
                #{s.chunk_id}
              </span>
            </div>
            <p>{s.preview}…</p>
          </li>
        ))}
      </ol>
    </section>
  );
}
