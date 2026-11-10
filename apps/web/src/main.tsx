import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { SarangWidget } from "./SarangWidget";
import "./styles.css";

/**
 * Standalone surface.
 *
 * The same component is what the portfolio embeds; this page exists so the
 * gateway can serve the whole product from one origin while the widget is being
 * developed and benchmarked.
 */
function Standalone() {
  return (
    <main className="sarang-standalone">
      <header className="sarang-standalone-head">
        <span className="sarang-standalone-mark" aria-hidden="true">
          S
        </span>
        <div>
          <h1>Sara</h1>
          <p>Sanket Sonwane&apos;s voice agent — his work, background and fit for a role.</p>
        </div>
      </header>

      <SarangWidget mode="inline" />

      <footer className="sarang-standalone-foot">
        <p>
          Voice runs over WebRTC where available and WebSocket PCM otherwise. Every answer comes from a curated fact
          base; nothing here invents a project, a metric or a claim.
        </p>
        <p>
          Microphone access starts only on your click. Keyboard and text-only use are fully supported.
        </p>
      </footer>
    </main>
  );
}

const container = document.getElementById("root");
if (container) {
  createRoot(container).render(
    <StrictMode>
      <Standalone />
    </StrictMode>,
  );
}
