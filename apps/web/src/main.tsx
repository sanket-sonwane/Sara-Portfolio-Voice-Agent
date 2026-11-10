import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { SarangWidget } from "./SarangWidget";
import "./styles.css";

type Theme = "light" | "dark";

const params = new URLSearchParams(window.location.search);
const embed = params.get("embed") === "1";
const themeParam = params.get("theme");
const theme: Theme | undefined = themeParam === "light" || themeParam === "dark" ? themeParam : undefined;

/**
 * Embedded surface.
 *
 * `?embed=1` drops the standalone header/footer so the widget is the whole
 * document — the portfolio supplies the panel chrome (frame, close button,
 * connecting overlay) and the widget supplies the conversation UI.
 */
function Embed() {
  return (
    <main className="sarang-embed-surface">
      <SarangWidget mode="inline" appearance="embed" theme={theme} />
    </main>
  );
}

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

if (embed) {
  document.documentElement.dataset.sarangTheme = theme ?? "light";
}

const container = document.getElementById("root");
if (container) {
  createRoot(container).render(<StrictMode>{embed ? <Embed /> : <Standalone />}</StrictMode>);
}
