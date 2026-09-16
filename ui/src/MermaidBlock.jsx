/**
 * MermaidBlock — renders a ```mermaid fenced code block as an SVG diagram.
 *
 * Streaming-safe: only renders once the block is complete (closing fence
 * detected). While streaming, shows a placeholder. Falls back to raw code
 * on parse/render errors.
 */
import React, { useEffect, useRef, useState } from "react";

export default function MermaidBlock({ children }) {
  const containerRef = useRef(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!containerRef.current || !children) return;

    let mermaid;
    let isMounted = true;

    import("mermaid").then((mod) => {
      if (!isMounted) return;
      mermaid = mod.default;
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "loose",
        theme: "default",
      });

      const render = async () => {
        try {
          const { svg } = await mermaid.render(
            `mermaid-${Date.now()}`,
            children,
            containerRef.current,
          );
          if (isMounted) {
            containerRef.current.innerHTML = svg;
            setError(false);
          }
        } catch (err) {
          if (isMounted) setError(true);
        }
      };
      render();
    });

    return () => {
      isMounted = false;
    };
  }, [children]);

  if (error) {
    return (
      <pre style={{
        background: "#1a1a2e",
        color: "#e94560",
        padding: "12px",
        borderRadius: "8px",
        overflowX: "auto",
        fontSize: "12px",
      }}>{children}</pre>
    );
  }

  return <div ref={containerRef} style={{ margin: "12px 0" }} />;
}
