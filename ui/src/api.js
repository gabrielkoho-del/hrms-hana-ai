/**
 * API client for the HR chat backend.
 *
 * Handles:
 * - Bearer token selection (employee vs admin) from window.__HR_TOKENS__
 * - X-HR-Client: custom-ui header (gates hr_chart side-channel)
 * - SSE streaming parsing with hr_progress and hr_chart side-channel support
 */

const API_BASE = "/v1";

function getToken() {
  const tokens = window.__HR_TOKENS__ || {};
  const role = window.__HR_ROLE__ || "employee";
  return tokens[role] || tokens.employee || "";
}

function getHeaders(extra = {}) {
  const headers = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${getToken()}`,
    "X-HR-Client": "custom-ui",
    ...extra,
  };
  return headers;
}

/**
 * Stream a chat completion.
 *
 * Yields events of the form:
 *   { type: "progress", stage, label, detail }
 *   { type: "chart", payload: {type, title, labels, datasets, options} }
 *   { type: "content", content: "chunk text" }
 *   { type: "done" }
 *   { type: "error", message: "friendly error" }
 */
export async function* streamChatCompletion(messages) {
  const res = await fetch(`${API_BASE}/chat/completions`, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({
      model: "hr-agent",
      stream: true,
      messages,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    yield { type: "error", message: err.detail || `HTTP ${res.status}` };
    return;
  }

  if (!res.body) {
    yield { type: "error", message: "No response body" };
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop(); // keep incomplete line

    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const payload = line.replace("data:", "").trim();
      if (payload === "[DONE]") {
        yield { type: "done" };
        continue;
      }
      try {
        const parsed = JSON.parse(payload);

        // Side-channel: structured chart payload (custom UI only)
        if (parsed.hr_chart) {
          yield { type: "chart", payload: parsed.hr_chart };
          continue;
        }

        // Side-channel: progress event
        if (parsed.hr_progress) {
          yield {
            type: "progress",
            stage: parsed.hr_progress.stage,
            label: parsed.hr_progress.label,
            detail: parsed.hr_progress.detail,
          };
          continue;
        }

        // Standard OpenAI content chunk
        const delta = parsed.choices?.[0]?.delta;
        if (delta?.content) {
          yield { type: "content", content: delta.content };
        }
        if (parsed.choices?.[0]?.finish_reason === "stop") {
          yield { type: "stop" };
        }
      } catch {
        // ignore malformed/partial frame
      }
    }
  }
}
