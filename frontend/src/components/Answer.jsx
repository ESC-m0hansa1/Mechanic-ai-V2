import React from "react";

// The model is instructed to cite like "(p. 693)". Turning those into buttons
// closes the loop: the citation the model produced points at the retrieved chunk
// it came from, so a reader can check the claim instead of trusting it.
const CITATION = /\((?:p\.?|page)\s*(\d+)\)/gi;

function linkifyCitations(text, onCite) {
  const parts = [];
  let cursor = 0;
  for (const match of text.matchAll(CITATION)) {
    if (match.index > cursor) parts.push(text.slice(cursor, match.index));
    const page = match[1];
    parts.push(
      <button
        key={`${match.index}-${page}`}
        type="button"
        className="cite"
        onClick={() => onCite(Number(page))}
        title={`Show the excerpt from page ${page}`}
      >
        p.{page}
      </button>,
    );
    cursor = match.index + match[0].length;
  }
  parts.push(text.slice(cursor));
  return parts;
}

export default function Answer({ result, onCite }) {
  const { answer, refused, retrieval_ms, total_ms, strategy } = result;

  return (
    <section className={`answer ${refused ? "answer-refused" : ""}`}>
      <div className="answer-head">
        <h2>{refused ? "Not in this manual" : "Answer"}</h2>
        <ul className="badges">
          <li title="Retrieval strategy that selected these excerpts">{strategy}</li>
          <li title="Time spent retrieving and reranking">retrieval {retrieval_ms} ms</li>
          <li title="Total round trip, including the language model">
            total {total_ms} ms
          </li>
        </ul>
      </div>

      {/* Paragraph splitting rather than a markdown renderer: the model returns
          short prose, and pulling in a markdown library to render two paragraphs
          would add a dependency and an XSS surface for no benefit. */}
      {answer.split(/\n{2,}/).map((para, i) => (
        <p key={i}>{linkifyCitations(para, onCite)}</p>
      ))}

      {refused && (
        <p className="note">
          The retrieved excerpts did not contain an answer, so the model was
          instructed to say so. A confident-sounding guess would be worse than
          nothing for a repair decision.
        </p>
      )}
    </section>
  );
}
