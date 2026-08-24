import type { RagMode } from "../api/types";

export function ModeToggle({ mode, onChange }: { mode: RagMode; onChange: (mode: RagMode) => void }) {
  return (
    <div className="mode-toggle" role="radiogroup" aria-label="RAG mode">
      <button
        type="button"
        role="radio"
        aria-checked={mode === "classic"}
        className={`mode-toggle__option ${mode === "classic" ? "is-active" : ""}`}
        onClick={() => onChange("classic")}
      >
        Classic RAG
      </button>
      <button
        type="button"
        role="radio"
        aria-checked={mode === "agent"}
        className={`mode-toggle__option ${mode === "agent" ? "is-active" : ""}`}
        onClick={() => onChange("agent")}
      >
        Agentic RAG
      </button>
    </div>
  );
}
