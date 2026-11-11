// Headless smoke test for the built widget, including the diagnostics drawer.
//
// A real-mic pass cannot be automated (a headless browser cannot be granted a
// microphone), but everything around it can: that the page mounts, that a typed
// turn produces a real metrics.turn, that the drawer renders both the empty and
// populated states, and that nothing throws. This is what found the first
// version of the session log calling a typed turn a "spoken turn".
//
//   node scripts/smoke.mjs                       # against localhost:8080
//   node scripts/smoke.mjs --url https://... --question "What did Sanket build?"
//
// Exits non-zero if the drawer does not open or the page logs an error.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
].filter(Boolean);

function parseArgs(argv) {
  const args = { url: "http://127.0.0.1:8080/", question: "", screenshot: "" };
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i].replace(/^--/, "");
    if (key in args) args[key] = argv[i + 1] ?? "";
  }
  return args;
}

const args = parseArgs(process.argv.slice(2));
const port = 9333 + Math.floor(Math.random() * 200);
const profile = join(tmpdir(), `sarang-smoke-${port}`);
const chromePath = CHROME_CANDIDATES.find((candidate) => existsSync(candidate));

if (!chromePath) {
  console.error("no Chrome found; set CHROME_PATH");
  process.exit(2);
}

rmSync(profile, { recursive: true, force: true });
mkdirSync(profile, { recursive: true });

const chrome = spawn(
  chromePath,
  [
    "--headless=new",
    "--disable-gpu",
    "--no-first-run",
    "--no-default-browser-check",
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profile}`,
    "--window-size=1280,1400",
    "--autoplay-policy=no-user-gesture-required",
    "about:blank",
  ],
  { stdio: "ignore" },
);

const cleanup = () => {
  chrome.kill();
  try {
    rmSync(profile, { recursive: true, force: true, maxRetries: 3, retryDelay: 150 });
  } catch {
    // Windows holds Crashpad files open for a moment after the process exits.
    // The profile lives in the temp directory, so leaving it is harmless and
    // must not turn a passing smoke test into a failure.
  }
};

let target = null;
for (let attempt = 0; attempt < 60 && !target; attempt += 1) {
  await sleep(250);
  try {
    const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    target = list.find((entry) => entry.type === "page") ?? null;
  } catch {
    // browser not listening yet
  }
}
if (!target) {
  cleanup();
  console.error("chrome did not expose a page target");
  process.exit(2);
}

const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  ws.addEventListener("open", resolve, { once: true });
  ws.addEventListener("error", reject, { once: true });
});

let nextId = 1;
const pending = new Map();
const errors = [];

ws.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    const { resolve, reject } = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) reject(new Error(JSON.stringify(message.error)));
    else resolve(message.result);
    return;
  }
  if (message.method === "Runtime.exceptionThrown") {
    const detail = message.params?.exceptionDetails;
    errors.push(detail?.exception?.description ?? detail?.text ?? "exception");
  }
  if (message.method === "Runtime.consoleAPICalled" && message.params?.type === "error") {
    errors.push(
      message.params.args.map((a) => a.value ?? a.description ?? "?").join(" "),
    );
  }
});

function send(method, params = {}) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
}

async function evaluate(expression) {
  const result = await send("Runtime.evaluate", {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.exception?.description ?? "evaluate failed");
  }
  return result.result?.value;
}

await send("Page.enable");
await send("Runtime.enable");
await send("Page.navigate", {
  url: `${args.url}${args.url.includes("?") ? "&" : "?"}smoke=${Date.now()}`,
});
await sleep(6000);

if (args.question) {
  const asked = await evaluate(`(() => {
    const input = [...document.querySelectorAll("input, textarea")]
      .find((node) => /type|ask/i.test(node.placeholder || ""));
    if (!input) return "no input";
    const proto = input.tagName === "TEXTAREA"
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, "value").set.call(input, ${JSON.stringify(args.question)});
    input.dispatchEvent(new Event("input", { bubbles: true }));
    const form = input.closest("form");
    if (form) { form.requestSubmit(); return "submitted"; }
    const button = [...document.querySelectorAll("button")].find((b) => b.type === "submit");
    if (button) { button.click(); return "submitted"; }
    return "no submit";
  })()`);
  console.log(`ask: ${asked}`);
  await sleep(16000);
}

const opened = await evaluate(`(() => {
  const button = [...document.querySelectorAll("button")]
    .find((b) => /diagnostics/i.test(b.textContent || ""));
  if (!button) return "no diagnostics button";
  button.click();
  return "opened";
})()`);
await sleep(900);

const drawer = await evaluate(`(() => {
  const node = document.querySelector(".sarang-diagnostics");
  if (!node) return null;
  return {
    sections: [...node.querySelectorAll("h4")].map((h) => h.textContent),
    session: node.querySelector("section:nth-of-type(2)")?.innerText ?? "",
  };
})()`);

if (args.screenshot) {
  const shot = await send("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: true,
  });
  writeFileSync(args.screenshot, Buffer.from(shot.data, "base64"));
  console.log(`screenshot: ${args.screenshot}`);
}

ws.close();
cleanup();

console.log(`drawer: ${opened}`);
console.log(`sections: ${drawer ? drawer.sections.join(" · ") : "MISSING"}`);
if (drawer) console.log(`session: ${drawer.session.replace(/\n+/g, " | ")}`);
console.log(`console errors: ${errors.length ? errors.join(" / ") : "none"}`);

const failed = !drawer || opened !== "opened" || errors.length > 0;
process.exit(failed ? 1 : 0);
