# Class-Node

A real-time quiz and classroom interaction system. Class-Node is built around a
simple constraint: every participant must see the same question at the same
time, and the host must be able to trust the live result.

## Real-time synchronisation

Quiz state is server-owned. A question has one authoritative state — waiting,
open, locked, revealed — and clients render that state rather than keeping their
own copy. When the host advances, the change propagates through a realtime
channel, and a late joiner is given the current state instead of a replay.

That design removes the classic race where two students answer against different
questions because their local timers drifted. The timer is a presentation of
server state, not the source of truth.

## Quiz generation

Question sets can be generated with AI, which makes authoring a quiz cheap. The
generated set is still reviewed and stored as ordinary quiz content, so a bad
generation is an editing problem rather than a live failure.

## Analytics

The host gets a live analytics view: participation, answer distribution and
per-question outcomes. Because answers are recorded against the authoritative
question state, the dashboard reflects the same data the scoring uses.

## Stack

React and TypeScript on the client, Node.js on the server and Supabase for
realtime data and persistence.

## Traction

The platform served more than one hundred weekly users. That number is the one
documented usage figure for the project; there are no published latency or
retention numbers, and the portfolio does not claim them.

## What it demonstrates

Real-time product engineering is mostly state discipline. Once the server owns
the state and clients render it, the hard bugs move from "why did the screens
disagree" to ordinary feature work.
