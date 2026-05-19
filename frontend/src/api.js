const API_BASE = import.meta.env.VITE_API_URL ?? "";

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) {
    throw new Error("Failed to reach the API");
  }
  return res.json();
}

export async function generateText({
  prompt,
  maxNewTokens,
  temperature,
  topK,
}) {
  const res = await fetch(`${API_BASE}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      max_new_tokens: maxNewTokens,
      temperature,
      top_k: topK,
    }),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => d.msg).join(", ")
          : "Generation failed";
    throw new Error(message);
  }
  return data;
}
