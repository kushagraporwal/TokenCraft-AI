import { useCallback, useEffect, useState } from "react";
import { fetchHealth, generateText } from "./api";
import "./App.css";

const DEFAULT_PROMPT = "The future of artificial intelligence is";

export default function App() {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState(null);

  const [maxNewTokens, setMaxNewTokens] = useState(80);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(50);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: "offline" }));
  }, []);

  const handleGenerate = useCallback(
    async (e) => {
      e.preventDefault();
      setError("");
      setLoading(true);
      setOutput("");

      try {
        const result = await generateText({
          prompt: prompt.trim(),
          maxNewTokens,
          temperature,
          topK,
        });
        setOutput(result.output);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    },
    [prompt, maxNewTokens, temperature, topK]
  );

  const checkpointLabel =
    health?.checkpoint_loaded === true
      ? "Trained weights loaded"
      : health?.checkpoint_loaded === false
        ? "No checkpoint — using random weights"
        : "Checking API…";

  return (
    <div className="app">
      <header className="header">
        <div>
          <p className="eyebrow">GPT-style LLM</p>
          <h1>Text generation</h1>
          <p className="subtitle">
            Enter a prompt and your model will continue the text.
          </p>
        </div>
        <div className={`status-pill ${health?.status === "ok" ? "ok" : ""}`}>
          <span className="status-dot" />
          {health?.status === "ok" ? checkpointLabel : "API offline"}
        </div>
      </header>

      <main className="layout">
        <form className="panel input-panel" onSubmit={handleGenerate}>
          <label htmlFor="prompt">Your input</label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Type a prompt for the model…"
            rows={6}
            disabled={loading}
          />

          <div className="controls">
            <label>
              Max tokens
              <input
                type="number"
                min={1}
                max={512}
                value={maxNewTokens}
                onChange={(e) => setMaxNewTokens(Number(e.target.value))}
                disabled={loading}
              />
            </label>
            <label>
              Temperature
              <input
                type="number"
                min={0}
                max={2}
                step={0.1}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                disabled={loading}
              />
            </label>
            <label>
              Top-k
              <input
                type="number"
                min={1}
                max={500}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                disabled={loading}
              />
            </label>
          </div>

          <button type="submit" disabled={loading || !prompt.trim()}>
            {loading ? "Generating…" : "Generate"}
          </button>

          {error && <p className="error">{error}</p>}
        </form>

        <section className="panel output-panel" aria-live="polite">
          <label>Model output</label>
          <div className="output-box">
            {loading && <p className="placeholder pulse">Thinking…</p>}
            {!loading && output && <pre>{output}</pre>}
            {!loading && !output && (
              <p className="placeholder">
                Generated text will appear here after you run the model.
              </p>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
