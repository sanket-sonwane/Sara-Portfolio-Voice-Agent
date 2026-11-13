"""The three text states of one turn.

They are deliberately separate values, never one string with a flag:

``generated``
    What the language model produced. It can be longer than anything the
    visitor heard, and it is the only value a retry may start from.

``committed``
    The segments TTS actually finished synthesising and handed to egress. A
    segment that failed halfway is *not* committed, even if half of its audio
    was already queued.

``audible``
    The prefix the browser confirms it actually played. It is reported by the
    client, never inferred on the server, because a frame that was submitted to
    the transport is not a frame the visitor heard.

Conflating them produced three recurring bugs: a transcript that claimed
sentences no one heard, a barge-in that "cancelled" text that had already been
spoken, and latency measured from generated text to a predicted playback time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TurnText:
    generated: str = ""
    # One entry per segment whose synthesis completed and whose audio reached
    # the egress. Order is the playback order.
    committed_segments: list[str] = field(default_factory=list)
    # Confirmed by the browser. Empty until it reports.
    audible: str = ""
    audible_chars: int = 0

    @property
    def committed(self) -> str:
        return " ".join(segment.strip() for segment in self.committed_segments if segment.strip())

    def commit(self, text: str) -> None:
        if text.strip():
            self.committed_segments.append(text.strip())

    def note_audible(self, text: str | None = None, *, chars: int | None = None) -> None:
        """Record what the browser says was played.

        ``chars`` is the length of the committed prefix that was audible, which
        is what the client can cheaply compute from playback progress. A client
        that reports text instead is accepted only while it is a prefix of the
        committed text; anything else would invent speech.
        """
        committed = self.committed
        if chars is not None:
            self.audible_chars = max(self.audible_chars, min(int(chars), len(committed)))
            self.audible = committed[: self.audible_chars]
            return
        if text is not None and committed.startswith(text):
            self.audible = text
            self.audible_chars = len(text)

    def transcript(self) -> str:
        """The honest transcript: what was spoken, falling back to generated.

        An interrupted turn publishes the committed prefix, not the generated
        answer. A turn whose TTS never started publishes nothing, which is the
        correct answer to "what did the visitor hear".
        """
        committed = self.committed
        if committed:
            return committed
        return self.generated.strip()

    def payload(self) -> dict:
        return {
            "generated_text": self.generated.strip(),
            "committed_text": self.committed,
            "audible_text": self.audible,
            "audible_chars": self.audible_chars,
        }
