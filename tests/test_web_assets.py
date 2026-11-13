"""Frontend assets that fail silently in the browser.

The PCM worklet is loaded by ``audioWorklet.addModule()`` directly: the browser
parses it as JavaScript, with no build step. A TypeScript annotation in that
file makes ``addModule()`` reject, and the old code swallowed that rejection and
fell back to the deprecated, main-thread ScriptProcessorNode. The pipeline kept
"working" while every latency number was measured on the wrong path.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKLET = ROOT / "apps" / "web" / "public" / "pcm-worklet.js"


def test_the_worklet_exists_and_registers_a_processor():
    source = WORKLET.read_text(encoding="utf-8")
    assert 'registerProcessor("sarang-pcm"' in source
    assert "AudioWorkletProcessor" in source


def test_the_worklet_has_no_type_annotations():
    source = WORKLET.read_text(encoding="utf-8")
    # The exact regression: a parameter annotation is a parse error here.
    assert "Float32Array[][]" not in source
    assert "process(inputs:" not in source


def test_the_worklet_parses_as_javascript():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    result = subprocess.run(
        [node, "--check", str(WORKLET)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


METRICS = ROOT / "apps" / "web" / "src" / "metrics.ts"


def test_client_eos_to_audible_ignores_unpaired_or_reversed_stamps():
    """The browser-side latency must be absent, not zero, when it is unknowable."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = """
const m = await import(process.argv[1]);
const eq = (left, right, label) => {
  if (!Object.is(left, right)) throw new Error(`${label}: ${left} !== ${right}`);
};
eq(m.clientEosToAudible(100, 420), 320, "paired");
eq(m.clientEosToAudible(100.4, 100.6), 0, "rounds to zero");
eq(m.clientEosToAudible(null, 420), undefined, "missing end of speech");
eq(m.clientEosToAudible(100, undefined), undefined, "missing audible stamp");
eq(m.clientEosToAudible(500, 100), undefined, "audible before speech end");
"""
    result = subprocess.run(
        [
            node,
            "--experimental-strip-types",
            "--input-type=module",
            "-e",
            script,
            METRICS.as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if "bad option" in result.stderr or "Unknown file extension" in result.stderr:
        pytest.skip("this node cannot strip TypeScript types")
    assert result.returncode == 0, result.stderr
