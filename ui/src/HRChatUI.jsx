/**
 * HRChatUI — Custom HR chat UI for employees.
 *
 * Wired to the FastAPI /v1/chat/completions SSE endpoint with:
 * - hr_progress side-channel for the status line
 * - hr_chart side-channel for native Chart.js rendering (X-HR-Client: custom-ui)
 * - react-markdown for tables, links, and mermaid fallback
 * - Role toggle that swaps JWT tokens (not model names)
 */
import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Send,
  Plus,
  MessageSquare,
  Sparkles,
  Calendar,
  Wallet,
  FileText,
  Users,
  Search,
  ClipboardCheck,
  BarChart3,
  UserCog,
  Settings,
} from "lucide-react";
import ChartCanvas from "./ChartCanvas";
import MermaidBlock from "./MermaidBlock";
import { streamChatCompletion } from "./api";

const COLORS = {
  bg: "#F7F6F1",
  paper: "#FFFFFF",
  sidebar: "#EFEDE5",
  ink: "#242A24",
  inkSoft: "#5C6259",
  border: "#E3E1D6",
  sage: "#3F5745",
  sageSoft: "#E7ECE4",
  sageDeep: "#2E4234",
};

const EMPLOYEE_SUGGESTIONS = [
  { icon: Calendar, text: "How many leave days do I have left?" },
  { icon: Wallet, text: "When is the next payroll date?" },
  { icon: FileText, text: "What's the work-from-home policy?" },
  { icon: Users, text: "Who's on leave this week?" },
];

const ADMIN_SUGGESTIONS = [
  { icon: ClipboardCheck, text: "What leave requests are pending my approval?" },
  { icon: Search, text: "Show Aisha Rahman's leave history" },
  { icon: BarChart3, text: "What's this quarter's headcount by department?" },
  { icon: UserCog, text: "Who has admin access to payroll?" },
];

const SEED_HISTORY = [
  { id: 1, title: "Leave balance question" },
  { id: 2, title: "Payslip for August" },
  { id: 3, title: "WFH policy clarification" },
];

function StatusLine({ progress }) {
  if (!progress) return null;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, color: COLORS.inkSoft, padding: "4px 0 4px 40px" }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: COLORS.sage, animation: "pulse 1.1s ease-in-out infinite" }} />
      {progress.label}
      <style>{`@keyframes pulse{0%,100%{opacity:.35}50%{opacity:1}}`}</style>
    </div>
  );
}

function MessageContent({ text, chart }) {
  // If we have a structured chart payload, render it natively.
  // The markdown text may still contain data tables and links.
  return (
    <>
      {chart && <ChartCanvas config={chart} />}
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code: ({ node, inline, ...props }) => {
            const language = node?.language || "";
            if (!inline && language === "mermaid") {
              return <MermaidBlock>{node.content}</MermaidBlock>;
            }
            return <code {...props} />;
          },
          img: ({ node, ...props }) => {
            // Render chart images (shouldn't arrive for custom-ui, but handle gracefully)
            return <img {...props} style={{ maxWidth: "100%", borderRadius: "8px", margin: "8px 0" }} />;
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </>
  );
}

export default function HRChatUIStandard() {
  const [role, setRole] = useState("employee");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [progress, setProgress] = useState(null);
  const [busy, setBusy] = useState(false);
  const [pendingChart, setPendingChart] = useState(null);
  const scrollRef = useRef(null);
  const isAdmin = role === "admin";
  const suggestions = isAdmin ? ADMIN_SUGGESTIONS : EMPLOYEE_SUGGESTIONS;

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, progress]);

  function newChat() {
    setMessages([]);
    setProgress(null);
    setBusy(false);
    setPendingChart(null);
  }

  async function send(text) {
    const question = text ?? input;
    if (!question.trim() || busy) return;

    setMessages((m) => [...m, { role: "user", text: question }]);
    setInput("");
    setBusy(true);
    setProgress({ stage: "thinking", label: "Thinking..." });
    setPendingChart(null);

    let assistantText = "";
    let assistantAdded = false;

    try {
      for await (const event of streamChatCompletion(
        [...messages, { role: "user", content: question }].map((m) => ({
          role: m.role,
          content: m.text ?? m.content,
        })),
      )) {
        if (event.type === "progress") {
          setProgress({ stage: event.stage, label: event.label, detail: event.detail });
        } else if (event.type === "chart") {
          setPendingChart(event.payload);
          setProgress({ stage: "charting", label: "Rendering chart..." });
        } else if (event.type === "content") {
          assistantText += event.content;
          setProgress(null);
          if (!assistantAdded) {
            assistantAdded = true;
            setMessages((m) => [...m, { role: "assistant", text: assistantText, chart: pendingChart }]);
          } else {
            setMessages((m) => {
              const copy = [...m];
              copy[copy.length - 1] = { role: "assistant", text: assistantText, chart: pendingChart };
              return copy;
            });
          }
        } else if (event.type === "done" || event.type === "stop") {
          setBusy(false);
          setProgress(null);
        } else if (event.type === "error") {
          setBusy(false);
          setProgress(null);
          setMessages((m) => [...m, { role: "assistant", text: event.message }]);
        }
      }
    } catch (err) {
      setBusy(false);
      setProgress(null);
      setMessages((m) => [...m, { role: "assistant", text: "I couldn't reach the HR assistant service. Please try again." }]);
    }
  }

  const empty = messages.length === 0;

  return (
    <div
      style={{
        display: "flex",
        height: "660px",
        maxWidth: 1040,
        margin: "0 auto",
        background: COLORS.paper,
        borderRadius: 16,
        border: `1px solid ${COLORS.border}`,
        overflow: "hidden",
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        color: COLORS.ink,
      }}
    >
      {/* Sidebar */}
      <div style={{ width: 260, flexShrink: 0, background: COLORS.sidebar, display: "flex", flexDirection: "column" }}>
        <div style={{ padding: 12 }}>
          <button
            onClick={newChat}
            style={{
              width: "100%",
              display: "flex",
              alignItems: "center",
              gap: 8,
              fontSize: 13,
              fontWeight: 600,
              padding: "9px 12px",
              borderRadius: 9,
              border: `1px solid ${COLORS.border}`,
              background: COLORS.paper,
              cursor: "pointer",
              color: COLORS.ink,
            }}
          >
            <Plus size={15} /> New chat
          </button>
        </div>

        <div style={{ flex: 1, overflowY: "auto", padding: "4px 12px" }}>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.4, color: COLORS.inkSoft, textTransform: "uppercase", margin: "10px 8px 6px" }}>
            Recent
          </div>
          {SEED_HISTORY.map((h) => (
            <div
              key={h.id}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                fontSize: 13,
                padding: "8px 8px",
                borderRadius: 8,
                color: COLORS.inkSoft,
                cursor: "pointer",
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              <MessageSquare size={14} style={{ flexShrink: 0 }} />
              {h.title}
            </div>
          ))}
        </div>

        <div style={{ padding: 12, borderTop: `1px solid ${COLORS.border}` }}>
          <div style={{ display: "flex", background: COLORS.paper, border: `1px solid ${COLORS.border}`, borderRadius: 9, padding: 3, marginBottom: 10 }}>
            {[{ key: "employee", label: "Employee" }, { key: "admin", label: "HR admin" }].map((opt) => (
              <button
                key={opt.key}
                onClick={() => { setRole(opt.key); window.__HR_ROLE__ = opt.key; }}
                style={{
                  flex: 1,
                  fontSize: 12,
                  fontWeight: 600,
                  padding: "6px 8px",
                  borderRadius: 7,
                  border: "none",
                  cursor: "pointer",
                  background: role === opt.key ? COLORS.sageDeep : "transparent",
                  color: role === opt.key ? "#FFFFFF" : COLORS.inkSoft,
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, padding: "6px 8px", color: COLORS.inkSoft }}>
            <div style={{ width: 24, height: 24, borderRadius: "50%", background: COLORS.sageDeep, color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700 }}>
              GT
            </div>
            Gabriel Tan
            <Settings size={14} style={{ marginLeft: "auto" }} />
          </div>
        </div>
      </div>

      {/* Main */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, position: "relative" }}>
        {empty ? (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 24 }}>
            <div style={{ width: 44, height: 44, borderRadius: 12, background: COLORS.sageSoft, display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 16 }}>
              <Sparkles size={20} color={COLORS.sageDeep} />
            </div>
            <div style={{ fontSize: 20, fontWeight: 600, marginBottom: 6 }}>
              {isAdmin ? "How can I help manage HR today?" : "How can I help with HR today?"}
            </div>
            <div style={{ fontSize: 13, color: COLORS.inkSoft, marginBottom: 28 }}>
              {isAdmin ? "Approvals, employee records, analytics and policies" : "Leave, payroll, policies and your team"}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, width: "100%", maxWidth: 560 }}>
              {suggestions.map((s) => (
                <button
                  key={s.text}
                  onClick={() => send(s.text)}
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 10,
                    textAlign: "left",
                    fontSize: 13,
                    lineHeight: 1.4,
                    padding: "12px 14px",
                    borderRadius: 12,
                    border: `1px solid ${COLORS.border}`,
                    background: COLORS.paper,
                    cursor: "pointer",
                    color: COLORS.ink,
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = COLORS.bg)}
                  onMouseLeave={(e) => (e.currentTarget.style.background = COLORS.paper)}
                >
                  <s.icon size={16} color={COLORS.sage} style={{ flexShrink: 0, marginTop: 1 }} />
                  {s.text}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div ref={scrollRef} style={{ flex: 1, overflowY: "auto" }}>
            <div style={{ maxWidth: 640, margin: "0 auto", padding: "24px 20px" }}>
              {messages.map((m, i) => (
                <div key={i} style={{ display: "flex", gap: 12, marginBottom: 20 }}>
                  <div
                    style={{
                      width: 28,
                      height: 28,
                      borderRadius: "50%",
                      flexShrink: 0,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 11,
                      fontWeight: 700,
                      background: m.role === "user" ? COLORS.ink : COLORS.sageSoft,
                      color: m.role === "user" ? "#fff" : COLORS.sageDeep,
                    }}
                  >
                    {m.role === "user" ? "GT" : <Sparkles size={14} />}
                  </div>
                  <div style={{ fontSize: 14, lineHeight: 1.6, paddingTop: 3 }}>
                    <MessageContent text={m.text} chart={m.chart} />
                  </div>
                </div>
              ))}
              <StatusLine progress={progress} />
            </div>
          </div>
        )}

        {/* Floating input */}
        <div style={{ padding: "16px 20px 20px", borderTop: empty ? "none" : `1px solid ${COLORS.border}` }}>
          <div style={{ maxWidth: 640, margin: "0 auto" }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                background: COLORS.bg,
                border: `1px solid ${COLORS.border}`,
                borderRadius: 24,
                padding: "8px 8px 8px 18px",
                boxShadow: "0 2px 8px rgba(36,42,36,0.04)",
              }}
            >
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && send()}
                placeholder={isAdmin ? "Ask about approvals, employees or reports" : "Message HR assistant"}
                style={{ flex: 1, border: "none", outline: "none", background: "transparent", fontSize: 14, color: COLORS.ink }}
              />
              <button
                onClick={() => send()}
                disabled={busy}
                style={{
                  width: 34,
                  height: 34,
                  borderRadius: "50%",
                  border: "none",
                  background: COLORS.sageDeep,
                  color: "#fff",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  cursor: busy ? "default" : "pointer",
                  opacity: busy ? 0.5 : 1,
                  flexShrink: 0,
                }}
              >
                <Send size={15} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
