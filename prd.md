# SARANG
## Product Requirements Document (PRD)

**Version:** 1.0  
**Date:** 9 November 2026  
**Product Type:** Browser-based, real-time conversational AI agent  
**Primary Platform:** Personal portfolio website  
**Initial Language:** English  
**Architecture:** Custom event-driven cascade architecture  
**Document Status:** Approved product and architecture direction; implementation and provider benchmarks pending

---

# 1. Executive Summary

## 1.1 Product Overview

Sarang is a custom-built, real-time, browser-based AI calling agent that allows visitors to interact with a portfolio through natural voice conversations.

A visitor can open the portfolio and invoke the agent by clicking:

**"Talk to Sarang"**

The browser requests microphone permission and establishes a persistent real-time audio session. The visitor can then speak naturally and ask questions about the portfolio owner's:

- Technical background and skills.
- Software engineering projects.
- AI/ML and LLM engineering experience.
- Research publications.
- Internships and professional experience.
- Final-year academic project.
- Technical architectures and implementation decisions.
- Open-source work and demonstrations.

Sarang processes the visitor's speech using streaming ASR, understands the request using an LLM, retrieves relevant portfolio knowledge or invokes tools when required, and generates natural speech through streaming TTS.

The interaction must resemble a natural phone conversation rather than a conventional chatbot.

## 1.2 Core Product Philosophy

Sarang is not simply a user interface that connects three AI APIs.

It is a custom real-time distributed voice system.

The application owns:

- Browser-side audio processing.
- Real-time media transport.
- Session management.
- Streaming audio ingestion.
- Endpointing and conversational turn management.
- Provider orchestration.
- Incremental text processing.
- Streaming audio playback.
- Barge-in and interruption handling.
- Concurrency and cancellation.
- Observability and performance optimization.

Specialized inference providers perform the neural inference tasks.

This separation gives the engineering team control over the system's latency, accuracy, reliability and extensibility.

## 1.3 Product Value

Sarang has two primary purposes.

**Purpose A — Portfolio interaction**

Make the portfolio more interactive and let recruiters, hiring managers, engineers and collaborators explore technical work through conversation.

**Purpose B — Technical demonstration**

Demonstrate the ability to design and engineer a production-oriented real-time conversational system involving distributed systems, audio processing, inference orchestration, streaming protocols and performance engineering.

---

# 2. Problem Statement

Traditional developer portfolios are static.

Visitors must manually navigate project sections, open repositories, read descriptions and infer the depth of the developer's experience.

This creates several problems:

1. Visitors may not have time to examine every project.
2. Technical descriptions may not answer specific questions.
3. Complex architectures are difficult to explain through short project descriptions.
4. Recruiters may struggle to identify the most relevant experience.
5. Static portfolio interfaces do not offer natural follow-up conversations.
6. Visitors cannot immediately ask for deeper explanations or clarification.

A conventional text chatbot improves accessibility but does not deliver the desired conversational experience.

Sarang addresses these problems with a voice-first interface that makes portfolio information accessible through natural conversation.

## 2.1 Product Hypothesis

A fast, accurate and interruptible browser-based voice agent can make a technical portfolio substantially easier to explore.

The product hypothesis is that visitors will be more likely to discover relevant technical work when they can ask conversational questions rather than manually navigate a large collection of documents and project pages.

The hypothesis must be validated through real visitor engagement and conversation analytics.

---

# 3. Product Vision

## 3.1 Vision Statement

Build a voice agent that feels like having a real-time technical conversation with the portfolio owner.

The visitor should not need to understand how speech recognition, language models, vector retrieval or speech synthesis work.

They should experience:

- Immediate responsiveness.
- Natural conversational flow.
- Accurate technical explanations.
- Context-aware follow-up answers.
- Interruption at any time.
- Minimal perceptible waiting.
- Clear recovery when something goes wrong.

## 3.2 Design Principles

### Principle 1 — Voice first

Voice is the primary interaction modality.

Text is supporting information, not the primary product experience.

### Principle 2 — Low latency

Every component must be measured and optimized independently.

### Principle 3 — Streaming everywhere

Audio, recognition results, LLM tokens and synthesized audio should be processed incrementally.

### Principle 4 — Interruptibility

Visitors must be able to stop Sarang immediately and start a new turn.

### Principle 5 — Grounded intelligence

Sarang must distinguish documented facts from assumptions and must not invent portfolio achievements.

### Principle 6 — Modular infrastructure

ASR, LLM and TTS providers must be replaceable without redesigning the rest of the application.

### Principle 7 — Graceful degradation

Provider failures and network interruptions must not cause the entire portfolio website to become unusable.

---

# 4. Goals and Non-Goals

## 4.1 Primary Goals

| ID | Goal |
|---|---|
| G-01 | Build a custom browser-based voice agent. |
| G-02 | Support natural, bidirectional voice conversations. |
| G-03 | Achieve an end-of-speech-to-first-audio latency target below 500 ms. |
| G-04 | Maintain English speech recognition quality with WER below 6%. |
| G-05 | Implement streaming speech recognition. |
| G-06 | Implement streaming LLM generation. |
| G-07 | Implement streaming neural TTS. |
| G-08 | Support immediate barge-in and interruption. |
| G-09 | Retrieve relevant portfolio knowledge. |
| G-10 | Execute controlled tools for complex questions. |
| G-11 | Measure p50, p95 and p99 latency. |
| G-12 | Provide a reliable visitor experience through browser infrastructure. |

## 4.2 Secondary Goals

- Support continuous multi-turn conversations.
- Maintain conversation context.
- Show live transcription.
- Provide optional developer diagnostics.
- Support multiple concurrent sessions.
- Enable a future custom voice.
- Allow additional languages in future versions.
- Provide robust error recovery.
- Maintain cost visibility per interaction.

## 4.3 Non-Goals for Initial Release

The initial release will not include:

- Telephony or PSTN calling.
- WhatsApp voice calling.
- Outbound automated calls.
- Native mobile applications.
- Video conversations.
- Face or emotion recognition.
- Hindi or multilingual recognition.
- Voice cloning.
- User account creation.
- Persistent long-term visitor memory.
- Autonomous code execution.
- Unrestricted web browsing.
- Arbitrary external API execution.
- A general-purpose personal assistant.

The system is a portfolio-focused voice agent.

---

# 5. Target Users and Personas

## 5.1 Primary Persona — Recruiter / Hiring Manager

**Objective:** Understand the portfolio owner's experience and suitability for a role.

Example questions:

- What technologies does the developer work with?
- Which AI projects have been implemented?
- Tell me about the internships.
- What experience is relevant to AI engineering?
- Which projects demonstrate backend engineering skills?

Expected behavior:

Sarang should provide concise, relevant answers and offer deeper explanations where appropriate.

## 5.2 Secondary Persona — Technical Engineer

**Objective:** Understand implementation details and engineering depth.

Example questions:

- Explain the VisionMitra architecture.
- How does the CPU scheduling project work?
- How did the rogue-node detection system identify attacks?
- What inference optimizations were implemented?
- Which distributed systems challenges were addressed?

Expected behavior:

Sarang should retrieve relevant documentation, understand technical terminology and provide structured explanations.

## 5.3 Secondary Persona — Researcher / Technical Reviewer

**Objective:** Explore research contributions and methodology.

Example questions:

- Explain the problem formulation of the research.
- What was the experimental methodology?
- Which metrics were evaluated?
- What are the limitations?
- How does the proposed method compare with existing work?

Expected behavior:

Answers must be based on available research documents, with uncertainties and limitations communicated honestly.

## 5.4 Secondary Persona — General Visitor

**Objective:** Explore the portfolio casually.

Example questions:

- What do you build?
- What are your strongest projects?
- What is the most interesting project?
- What are you currently working on?

Expected behavior:

Friendly, natural and accessible conversation without overwhelming the visitor with unnecessary technical depth.

---

# 6. User Experience and Interaction Design

## 6.1 Entry Point

The portfolio must expose a prominent voice interaction entry point.

Primary CTA:

**Talk to Sarang**

The button should be accessible without requiring the visitor to navigate to another page.

## 6.2 Initial Experience

Expected sequence:

1. Visitor opens the portfolio.
2. Sarang entry point is visible.
3. Visitor selects Talk to Sarang.
4. Browser requests microphone permission.
5. User approves microphone access.
6. Sarang initializes the real-time connection.
7. Connection status changes to connected.
8. Sarang greets the visitor.
9. Visitor begins speaking.

The microphone must never be accessed before an explicit user action.

## 6.3 Interface States

The interface must visually distinguish the following states.

| State | Description | UI behavior |
|---|---|---|
| IDLE | No active session | Display Talk to Sarang |
| PERMISSION_REQUIRED | Microphone permission needed | Explain permission request |
| CONNECTING | Session and WebRTC initialization | Show connecting animation |
| CONNECTED | Active session | Display ready indicator |
| LISTENING | User speech detected | Animate microphone waveform |
| PROCESSING | Processing current utterance | Display thinking animation |
| SPEAKING | Sarang is generating audio | Animate output waveform |
| TOOL_WORKING | External operation in progress | Explain that additional information is being retrieved |
| INTERRUPTED | Current response cancelled | Transition back to listening |
| RECONNECTING | Network recovery underway | Display reconnection status |
| ERROR | Unrecoverable error | Display recovery action |
| ENDED | Session closed | Offer restart |

## 6.4 Conversational Experience

Sarang must:

- Greet visitors naturally.
- Avoid robotic introductions.
- Keep ordinary answers concise.
- Explain technical topics at an appropriate depth.
- Recognize follow-up questions.
- Avoid repeating portfolio information unnecessarily.
- Ask for clarification only when genuinely required.
- Use conversational transitions.
- Handle incomplete questions.
- Respond gracefully when information is unavailable.

## 6.5 Interruption Experience

Barge-in is a mandatory capability.

Example:

Sarang: "The VisionMitra project uses OpenStreetMap data and—"

Visitor: "Wait, explain the navigation part."

Expected behavior:

1. New speech is detected.
2. Browser playback stops immediately.
3. Existing audio queue is invalidated.
4. The active TTS generation is cancelled.
5. LLM generation is cancelled where supported.
6. The new utterance is transcribed.
7. Sarang processes the new request.
8. Conversation context is preserved.

The visitor must not hear old and new responses overlapping.

## 6.6 Live Transcript

Provide optional live captions.

The transcript should distinguish:

- Partial user transcription.
- Final user transcription.
- Sarang's generated response.

Partial transcription may be corrected or revised before finalization.

The interface must avoid presenting a changing partial transcript as a confirmed final statement.

## 6.7 Accessibility

The interface should:

- Support keyboard activation.
- Clearly indicate microphone status.
- Provide accessible labels.
- Respect reduced-motion settings.
- Offer captions.
- Provide a visible session-end control.
- Work at common desktop and mobile browser widths.
- Communicate permission and connection failures through text, not animation alone.

---

# 7. System Architecture

## 7.1 Architectural Style

Custom, modular, event-driven cascade architecture.

The product uses a cascade pipeline:

**Speech input → ASR → conversation engine → LLM → TTS → speech output**

Each inference component operates independently behind a provider adapter.

The system uses streaming execution instead of waiting for the complete output of each stage.

## 7.2 High-Level Architecture

```text
                      PORTFOLIO WEBSITE
                              |
                              v
                      TALK TO SARANG
                              |
                              v
                       BROWSER CLIENT
                              |
              +---------------+---------------+
              |                               |
              v                               v
        WebRTC Media                    WebSocket Control
          Channel                          Channel
              |                               |
              +---------------+---------------+
                              |
                              v
                       REAL-TIME GATEWAY
                         FASTAPI + ASYNCIO
                              |
              +---------------+---------------+
              |                               |
              v                               v
       AUDIO INGRESS                    AUDIO EGRESS
              |                               ^
              v                               |
       AUDIO PROCESSING                      |
              |                               |
         VAD + ENDPOINTING                    |
              |                               |
              v                               |
       NVIDIA NEMOTRON ASR                    |
              |                               |
              v                               |
       TRANSCRIPT / TURN MANAGER              |
              |                               |
              v                               |
       AGENT ORCHESTRATION                    |
              |                               |
         +----+----+                           |
         |         |                           |
         v         v                           |
       GROQ      PORTFOLIO                     |
       LLM         RAG                         |
         |         |                           |
         +----+----+                           |
              |                               |
              v                               |
       RESPONSE STREAM                        |
              |                               |
              v                               |
       TEXT SEGMENTER                         |
              |                               |
              v                               |
       CARTESIA SONIC                          |
              |                               |
              v                               |
       AUDIO CHUNK MANAGER                    |
              +-------------------------------+
```

## 7.3 Core Subsystems

### A. Browser Runtime

Responsible for:

- Microphone acquisition.
- WebRTC peer connection.
- Audio capture lifecycle.
- Audio playback.
- Local audio metering.
- UI state management.
- Interruption initiation.
- Connection statistics.

### B. Signaling Gateway

Responsible for:

- Session creation.
- WebSocket authentication.
- SDP exchange.
- ICE exchange.
- Connection lifecycle.
- Session-level control messages.

### C. Media Gateway

Responsible for:

- WebRTC media termination.
- Incoming audio reception.
- Audio decoding.
- Audio frame handling.
- Resampling.
- Output audio transmission.
- Media transport monitoring.

### D. Speech Engine

Responsible for:

- Voice activity detection.
- Endpointing.
- Streaming ASR.
- Partial transcript handling.
- Final transcript stabilization.
- Audio-to-text timing instrumentation.

### E. Conversation Engine

Responsible for:

- Turn identification.
- Conversation state.
- Context management.
- Barge-in coordination.
- Cancellation.
- Dialogue continuity.
- Session isolation.

### F. Agent Orchestration

Responsible for:

- Request classification.
- Fast-path response selection.
- Knowledge retrieval.
- Tool selection.
- Tool execution.
- Response synthesis.
- Evidence-aware answer generation.

### G. Speech Output Engine

Responsible for:

- LLM token ingestion.
- Incremental text segmentation.
- Text normalization.
- Streaming TTS.
- Audio chunk ordering.
- Audio format conversion.
- Playback coordination.

### H. Observability

Responsible for:

- Distributed tracing.
- Latency instrumentation.
- Error tracking.
- Usage measurement.
- Quality measurement.
- Performance dashboards.

---

# 8. Technology and Provider Decisions

## 8.1 Primary Stack

| Layer | Technology | Role |
|---|---|---|
| Frontend | React + TypeScript | User interface |
| Capture | Web Audio API | Browser audio handling |
| Media transport | WebRTC | Bidirectional real-time media |
| Signaling | WebSocket | Session and connection control |
| Backend | Python + FastAPI | Application gateway |
| Async runtime | asyncio | Concurrent task coordination |
| WebRTC server | aiortc initially | Python media gateway |
| STT | NVIDIA Nemotron ASR Streaming | Streaming English ASR |
| LLM | GroqCloud | Streaming reasoning and generation |
| TTS | Cartesia Sonic | Streaming speech synthesis |
| Database | PostgreSQL | Curated portfolio knowledge |
| Vector retrieval | pgvector | Semantic retrieval |
| Cache | Redis, when justified | Distributed session and cache state |
| Monitoring | OpenTelemetry | Distributed tracing |
| Metrics | Prometheus-compatible metrics | Operational measurements |
| Deployment | Docker | Reproducible services |

## 8.2 Provider Selection

### ASR — NVIDIA Nemotron

Selected provider: NVIDIA.

Initial candidate: Nemotron 3 ASR Streaming, English profile.

Model reference:

`nvidia/nemotron-speech-streaming-en-0.6b`

The model's cache-aware streaming architecture makes it relevant to low-latency incremental recognition.

Initial input chunk benchmark:

- 80 ms.
- 160 ms.
- 320 ms.

80 ms is the initial experimental profile.

It must not be treated as a guaranteed end-to-end latency.

The application will integrate through a cloud-hosted NVIDIA streaming endpoint using the supported speech-service protocol.

The provider adapter must isolate authentication, endpoint configuration, connection management and transcription events.

### LLM — Groq

Selected provider: GroqCloud.

Requirements:

- Streaming completion API.
- Low TTFT.
- Function/tool calling.
- Sufficient reasoning quality.
- Stable production availability.
- Predictable latency under expected concurrency.

Model selection must be benchmark-driven.

Do not permanently hardcode a model based on today's advertised token speed.

Compare:

1. Small, fast instruction models.
2. Larger, higher-quality instruction models.
3. Tool-capable models.

The fast path should prioritize low TTFT and reliable instruction following.

Complex technical investigations may use a stronger model where latency requirements permit.

### TTS — Cartesia Sonic

Selected provider: Cartesia.

Initial model family: Sonic.

Integration approach:

- Persistent WebSocket connection where supported.
- Incremental transcript submission.
- Context-aware generation.
- Chunked audio responses.
- Explicit audio context cancellation.
- Flush boundaries for managing streamed text segments.
- Configurable voice selection.

The exact model version and voice ID will be set through configuration rather than hardcoded throughout the application.

### Voice Identity

Custom voice cloning is deferred.

The first release uses a standard supported Cartesia voice.

Voice selection will be based on:

- English pronunciation.
- Natural pacing.
- Clarity.
- Conversational tone.
- Technical term pronunciation.
- Streaming responsiveness.

A custom voice can be introduced later with appropriate speaker consent.

---

# 9. Audio and Media Pipeline

## 9.1 Capture

The browser acquires microphone input only after explicit permission.

Requirements:

- Mono speech capture.
- Echo cancellation where supported.
- Noise suppression where supported.
- Automatic gain control where supported.
- Track lifecycle management.
- Microphone device-change handling.

The browser must not assume that requested device settings are necessarily the actual negotiated settings.

## 9.2 WebRTC Transport

WebRTC is the media plane.

Requirements:

- Bidirectional audio media.
- Opus negotiation where supported.
- ICE connectivity.
- STUN/TURN support.
- DTLS-SRTP media protection.
- Connection state monitoring.
- RTP statistics.
- Controlled reconnection.

WebSocket is the separate signaling and session-control plane.

The system must not transmit the primary media stream through ordinary JSON signaling messages.

## 9.3 Audio Frame Processing

The gateway receives decoded WebRTC audio frames.

The system converts and normalizes them into the format required by the ASR provider.

Initial ASR input profile:

- Sample rate: 16 kHz.
- Channels: Mono.
- Encoding: PCM signed 16-bit.
- Initial ASR aggregation window: 80 ms.

At 16 kHz mono PCM16:

- 20 ms = 640 bytes.
- 40 ms = 1,280 bytes.
- 80 ms = 2,560 bytes.
- 100 ms = 3,200 bytes.

The WebRTC RTP packetization interval is not the same thing as the ASR chunk aggregation interval.

The application must not introduce unnecessary capture-side waiting simply to create a fixed-size audio block.

## 9.4 Audio Queue Management

All audio queues must be bounded.

Required controls:

- Maximum queue length.
- Queue age.
- Drop policy.
- Backpressure handling.
- Generation identification.
- Session identification.
- Sequence numbering.

Audio from a cancelled generation must never be played after a newer generation has started.

---

# 10. Speech Recognition Requirements

## 10.1 Recognition Pipeline

```text
Browser Microphone
       |
       v
WebRTC Audio
       |
       v
Backend Media Gateway
       |
       v
Audio Decoder
       |
       v
Resampling / Normalization
       |
       v
VAD + Streaming Aggregator
       |
       v
NVIDIA Nemotron ASR
       |
       v
Partial Transcripts
       |
       v
Transcript Stabilization
       |
       v
Endpointing Decision
       |
       v
Final Transcript
```

## 10.2 Functional Requirements

**ASR-FR-01:** The system must stream audio continuously while the session is active.

**ASR-FR-02:** The recognizer must produce partial transcripts.

**ASR-FR-03:** Partial hypotheses must be treated as revisable.

**ASR-FR-04:** The recognizer must emit a finalized transcript for each committed user turn.

**ASR-FR-05:** The recognizer must support the English language configuration.

**ASR-FR-06:** The system must preserve punctuation and capitalization where provided by the model.

**ASR-FR-07:** The system must retain timing information where available.

**ASR-FR-08:** ASR must support persistent streaming sessions.

**ASR-FR-09:** A provider connection failure must be detected and reported.

**ASR-FR-10:** Final transcripts must be associated with the correct session and turn.

## 10.3 Recognition Quality

Target:

**Word Error Rate below 6%.**

WER:

WER = (Substitutions + Deletions + Insertions) / Reference Words

Evaluation must use independently verified reference transcripts.

Test categories:

- Ordinary English.
- Indian English accents.
- Technical vocabulary.
- Project names.
- Product and framework names.
- Fast speech.
- Slow speech.
- Speech with pauses.
- Noisy environments.
- Different microphones.

A separate technical vocabulary list should include terms from the portfolio.

Examples:

- VisionMitra.
- Nemotron.
- Cartesia.
- Groq.
- FastAPI.
- WebRTC.
- sched_ext.
- Kubernetes.
- pgvector.
- Machine learning.
- Context-aware CPU scheduling.

Where provider-side vocabulary biasing is supported, evaluate it. Otherwise, apply carefully constrained transcript normalization without silently changing the meaning.

---

# 11. VAD and Endpointing

## 11.1 Purpose

ASR determines what was spoken.

VAD determines whether speech is occurring.

Endpointing determines when the visitor's turn has ended.

These are separate responsibilities.

## 11.2 Initial Architecture

Use a lightweight VAD implementation with an adaptive endpointing controller.

Initial experiments:

| Parameter | Benchmark range |
|---|---|
| VAD frame | 10–30 ms |
| Speech onset confirmation | 40–80 ms |
| Endpoint silence | 100–220 ms |
| Longer pause allowance | Up to 300 ms |
| Minimum utterance duration | Configurable |

These values are experimental starting points.

We must evaluate the effect of endpointing thresholds on both response latency and premature turn termination.

## 11.3 Endpointing Requirements

- Detect speech onset.
- Detect silence.
- Distinguish short pauses from completed turns.
- Avoid cutting off sentences.
- Handle trailing words.
- Support adaptive thresholds.
- Work under moderate background noise.
- Handle interruptions during agent playback.

The system must record endpointing delay separately from ASR finalization delay.

---

# 12. Conversation Engine

## 12.1 Responsibility

The conversation engine coordinates all operations that make the system feel like a continuous conversation.

It owns:

- Turn lifecycle.
- Conversation state.
- Response generation lifecycle.
- Tool execution state.
- Context continuity.
- Interruption.
- Cancellation.
- Response ordering.

It must not be implemented as a collection of loosely synchronized UI flags.

## 12.2 State Machine

```text
IDLE
 |
 v
CONNECTING
 |
 v
LISTENING
 |
 v
TRANSCRIBING
 |
 v
THINKING
 |
 v
SPEAKING
 |
 +-----------> LISTENING
 |
 +-----------> INTERRUPTED
                  |
                  v
               LISTENING

THINKING
 |
 +-----------> TOOL_EXECUTION
                    |
                    v
                  THINKING

ANY ACTIVE STATE
 |
 +-----------> RECONNECTING
 |
 +-----------> ERROR
 |
 +-----------> ENDED
```

## 12.3 Turn and Generation IDs

Every conversational turn must have a unique turn ID.

Every generated response must have a generation ID.

Example:

```json
{
  "session_id": "session_123",
  "turn_id": 12,
  "generation_id": 4,
  "event_type": "tts.audio_chunk",
  "sequence": 0
}
```

If generation 4 is cancelled and generation 5 begins, any late generation-4 audio or tokens must be rejected.

This rule applies to:

- LLM token streams.
- Tool results.
- TTS audio.
- Playback scheduling.

## 12.4 Barge-In Requirements

**CONV-FR-01:** Visitors can interrupt Sarang while speaking.

**CONV-FR-02:** Browser playback must stop immediately on confirmed barge-in.

**CONV-FR-03:** The old playback queue must be invalidated.

**CONV-FR-04:** The TTS provider generation must be cancelled.

**CONV-FR-05:** LLM generation must be cancelled where supported.

**CONV-FR-06:** Old generation events must be rejected.

**CONV-FR-07:** The conversation context must remain valid.

**CONV-FR-08:** The system must return to listening state.

**CONV-FR-09:** Accidental interruptions caused by echo must be minimized.

**CONV-FR-10:** Interruption latency must be measured.

---

# 13. LLM and Agent Orchestration

## 13.1 Core Responsibility

The LLM converts a user's finalized spoken request into an accurate, useful and conversational response.

Sarang must understand its identity, scope, portfolio knowledge and available tools.

The agent must remain focused on the portfolio and its technical work.

## 13.2 Response Paths

Sarang uses two principal execution paths.

### Fast Path — Direct Answers

Suitable for:

- Background questions.
- Skills.
- Technologies.
- Project summaries.
- Internship details.
- Known project architecture.
- Simple technical explanations.

Flow:

```text
Final Transcript
      |
      v
Intent Classification
      |
      v
Relevant Context
      |
      v
Groq Streaming LLM
      |
      v
Text Segmenter
      |
      v
Cartesia TTS
```

The system should not initiate an unnecessary tool call for information already available in the curated portfolio context.

### Tool Path — Complex Requests

Suitable for:

- Detailed technical explanations.
- Multi-document research.
- Architecture comparisons.
- Methodology retrieval.
- Requests requiring several pieces of evidence.

Flow:

```text
Final Transcript
      |
      v
Request Classification
      |
      v
Tool Selection
      |
      v
Tool Execution
      |
      v
Evidence Collection
      |
      v
LLM Synthesis
      |
      v
Streaming TTS
```

The system must never present an unfinished tool result as verified information.

For slower tasks, Sarang may briefly acknowledge the request while work continues.

The acknowledgement must not contain invented factual conclusions.

## 13.3 LLM Functional Requirements

**LLM-FR-01:** Streaming token generation.

**LLM-FR-02:** Configurable model selection.

**LLM-FR-03:** Conversation context management.

**LLM-FR-04:** Tool calling.

**LLM-FR-05:** Structured tool outputs.

**LLM-FR-06:** Portfolio-grounded answers.

**LLM-FR-07:** Concise conversational output.

**LLM-FR-08:** Graceful uncertainty handling.

**LLM-FR-09:** Cancellation support.

**LLM-FR-10:** Model and prompt version tracking.

## 13.4 System Prompt Requirements

The system prompt should define:

- Agent identity.
- Portfolio role.
- Communication style.
- Language.
- Grounding rules.
- Technical explanation behavior.
- Response length.
- Tool selection rules.
- Refusal boundaries.
- Privacy boundaries.
- Uncertainty behavior.

The system prompt must be versioned.

## 13.5 Context Management

Do not resend the entire conversation and complete portfolio documentation for every request.

Use:

1. Stable system prompt.
2. Compact recent conversation.
3. Rolling conversation summary when needed.
4. Relevant retrieved project context.
5. Selected tool definitions.

Maintain a token budget.

Avoid unnecessary history growth.

For long conversations, summarize older exchanges without losing important project or question context.

---

# 14. Portfolio Knowledge Base and RAG

## 14.1 Purpose

The portfolio knowledge base provides reliable technical information to Sarang.

It must be based on curated documents rather than unrestricted inference.

## 14.2 Information Categories

Initial knowledge domains:

| Domain | Example content |
|---|---|
| Professional profile | Skills, background, education |
| Projects | Architecture, implementation, features |
| AI and ML | Models, inference, RAG, evaluation |
| Systems | Operating systems, Linux scheduling |
| Security | SDN, MEC and rogue-node detection |
| Research | Publications and methodologies |
| Internships | Tasks, contributions, technologies |
| Events | Technical presentations and community involvement |

## 14.3 Knowledge Sources

Potential sources:

- Approved portfolio content.
- Resume.
- Project READMEs.
- Architecture documentation.
- Research papers.
- Verified technical reports.
- Public GitHub repositories.
- Explicitly approved project notes.

Sensitive private information must not be ingested automatically.

## 14.4 RAG Architecture

```text
CURATED DOCUMENTS
       |
       v
DOCUMENT INGESTION
       |
       v
PARSING AND CLEANING
       |
       v
CHUNKING
       |
       v
EMBEDDING
       |
       v
POSTGRESQL + PGVECTOR
       |
       v
RETRIEVAL
       |
       v
RELEVANCE FILTERING
       |
       v
LLM CONTEXT
       |
       v
GROUNDED ANSWER
```

## 14.5 Retrieval Requirements

- Maintain document identifiers.
- Preserve source metadata.
- Preserve section and page information where possible.
- Use document-level relevance.
- Limit irrelevant context.
- Support project filtering.
- Support metadata-based filtering.
- Support updates and deletion.
- Track the version of indexed documents.

## 14.6 Knowledge Quality

Every important portfolio claim should have an approved source.

Sarang must not:

- Invent performance metrics.
- Claim publications were accepted if they were not.
- Inflate project contributions.
- Invent employment history.
- Fabricate project results.
- Confuse proposed work with completed work.

When evidence is unavailable, the correct behavior is to communicate the limitation.

---

# 15. Tool System

## 15.1 Tool Architecture

Tools execute specific, approved operations.

They must be exposed through a controlled tool registry.

## 15.2 Initial Tools

| Tool | Purpose |
|---|---|
| search_portfolio | Search approved portfolio documents |
| retrieve_project | Retrieve specific project information |
| retrieve_project_architecture | Obtain technical architecture |
| retrieve_research | Retrieve publication and methodology content |
| retrieve_experience | Retrieve internship and professional experience |
| list_skills | Retrieve categorized technical skills |
| get_project_links | Retrieve approved public project links |

Additional tools can be introduced later.

## 15.3 Tool Requirements

- Explicit input schemas.
- Input validation.
- Execution timeout.
- Permission checks.
- Result size limits.
- Structured output.
- Error handling.
- Cancellation where feasible.
- Trace and turn identifiers.
- Tool execution logging without sensitive payloads.

## 15.4 Tool Security

The LLM must not be allowed to execute arbitrary Python, shell commands or unrestricted external requests.

Only approved tools may be invoked.

Retrieved documents are data, not system instructions.

---

# 16. Streaming TTS and Speech Output

## 16.1 TTS Architecture

```text
Groq Token Stream
       |
       v
Text Buffer
       |
       v
Sentence / Phrase Detection
       |
       v
Text Normalization
       |
       v
Cartesia Sonic WebSocket
       |
       v
Audio Chunk Responses
       |
       v
Chunk Ordering
       |
       v
Audio Format Conversion
       |
       v
WebRTC Outgoing Audio Track
       |
       v
Browser Playback
```

## 16.2 Text Segmentation

Do not wait for the entire LLM response.

The text segmenter should release a suitable speech segment when a natural boundary is available.

Initial segmentation parameters to benchmark:

- Approximately 8–20 words per segment.
- Natural punctuation boundaries.
- Small maximum waiting time for a first segment.
- Configurable minimum and maximum character counts.
- Safe handling of abbreviations and technical terms.

Do not send every token as a separate TTS request.

## 16.3 TTS Requirements

**TTS-FR-01:** Streaming speech generation.

**TTS-FR-02:** Configurable voice and model.

**TTS-FR-03:** Incremental audio delivery.

**TTS-FR-04:** Context-aware speech generation.

**TTS-FR-05:** Context cancellation.

**TTS-FR-06:** Audio chunk ordering.

**TTS-FR-07:** Correct sample-rate conversion.

**TTS-FR-08:** Consistent pronunciation.

**TTS-FR-09:** Text-to-audio sequence mapping.

**TTS-FR-10:** Graceful provider error handling.

## 16.4 Output Audio Handling

Each audio chunk must include:

- Session ID.
- Turn ID.
- Generation ID.
- Chunk sequence.
- Audio format.
- Sample rate.
- Timestamp.
- Payload.

The playback engine must reject stale chunks.

The first audio chunk must not be delayed until the entire response has been generated.

## 16.5 Speech Quality

Evaluate:

- Pronunciation.
- Naturalness.
- Conversational pace.
- Pause placement.
- Technical vocabulary.
- Sentence continuity.
- Audio artifacts.
- Segment transitions.

The goal is a natural conversational voice, not merely intelligible synthesized speech.

---

# 17. Real-Time Latency Requirements

This is one of the most important parts of the product specification.

## 17.1 Primary Target

**Target: Less than 500 milliseconds from the actual end of the visitor's spoken utterance to the first audible portion of Sarang's response.**

This is the user-experienced metric.

It must include:

- Endpointing.
- Final transcript resolution.
- Conversation routing.
- LLM first-token delay.
- Text segmentation.
- TTS first-audio delay.
- Network delivery.
- Browser playback scheduling.

We must not present server-side model latency alone as total latency.

## 17.2 Component Targets

| Component | Metric | Target |
|---|---|---:|
| VAD / Endpointing | End-of-speech detection | 80–150 ms initial target |
| ASR | Final transcript after endpoint decision | Below 100 ms |
| ASR | Word Error Rate | Below 6% |
| LLM | TTFT after accepted request | 200–300 ms |
| TTS | TTFA after first segment submission | Below 120 ms |
| TTS | Real-Time Factor | Below 1.0 |
| Playback | Scheduling / buffering | Below 40 ms |
| Full pipeline | End-of-speech to first audible response | Below 500 ms target |

These are engineering objectives, not guaranteed provider specifications.

## 17.3 Important Measurement Definitions

**End-of-speech:** The actual last speech sample in the user's utterance, established through annotated benchmark audio or equivalent measurement.

**Endpointing decision:** The system's decision that the user has completed the turn.

**ASR finalization latency:** Time between endpointing decision and finalized transcript availability.

**LLM TTFT:** Time from accepted LLM request submission to the first generated token received by the application.

**TTS TTFA:** Time from submission of the first valid text segment to receipt of the first usable audio chunk.

**First audible response:** The browser begins scheduling and rendering the first valid audio samples to the output device.

The browser playback-start event is an application-level proxy; true acoustic output latency requires external loopback measurement in controlled testing.

**RTF:**

RTF = Audio Generation Wall Time / Generated Audio Duration

An RTF below 1.0 indicates that speech is generated faster than its playback duration.

## 17.4 Percentile Targets

| Metric | Initial engineering ambition |
|---|---:|
| End-of-speech-to-audio p50 | Below 500 ms |
| End-of-speech-to-audio p95 | Below 800 ms |
| End-of-speech-to-audio p99 | Below 1,200 ms |
| Endpoint-to-audio p50 | Below 400 ms |
| Endpoint-to-audio p95 | Below 650 ms |

A p95 goal is necessary because a low average can hide poor experiences.

The under-500 ms objective must be validated under clearly documented network, concurrency and device conditions.

## 17.5 Latency Optimization Techniques

- Persistent ASR connections.
- Persistent TTS connections.
- Small streaming ASR chunks.
- Bounded audio queues.
- Incremental transcript processing.
- Compact prompts.
- Fast-path responses.
- Context reuse where supported.
- Streaming LLM output.
- Adaptive sentence segmentation.
- Immediate first-chunk playback.
- Minimal gateway processing.
- Correct regional placement.
- Prompt caching where supported.
- Cancellation of obsolete work.
- Avoidance of unnecessary RAG.

## 17.6 Speculative Execution

A later optimization may use stable partial ASR hypotheses to prepare retrieval or speculative LLM work before the final transcript is committed.

However:

- Speculative results must not be spoken as facts.
- Finalized transcript must validate the request.
- Obsolete speculative generations must be cancelled.
- Wasted inference must be measured.
- Incorrect partial hypotheses must not contaminate conversation state.

This is optional optimization, not a prerequisite for the first release.

---

# 18. Concurrency and Distributed Systems Requirements

## 18.1 Event-Driven Execution

The backend must use asynchronous event-driven coordination.

Separate worker responsibilities should include:

- WebRTC media reception.
- Audio frame normalization.
- VAD.
- ASR sending.
- ASR receiving.
- Transcript event handling.
- Turn management.
- LLM streaming.
- Tool execution.
- TTS streaming.
- Audio transmission.
- Metrics collection.

These operations should overlap when dependencies permit.

## 18.2 Event Envelope

Every internal event should contain:

```json
{
  "event_id": "uuid",
  "session_id": "session_123",
  "turn_id": 12,
  "generation_id": 4,
  "event_type": "transcript.final",
  "timestamp": "2026-11-09T10:00:00Z",
  "payload": {}
}
```

## 18.3 Event Categories

```text
audio.received
audio.processed
speech.started
speech.ended
transcript.partial
transcript.final
turn.committed
llm.started
llm.token
llm.completed
tool.started
tool.completed
tts.started
tts.audio_chunk
tts.completed
playback.started
playback.interrupted
generation.cancelled
turn.completed
session.error
```

## 18.4 Backpressure

The system must handle situations where producers generate data faster than consumers can process it.

Requirements:

- Bounded queues.
- Explicit queue capacity.
- Queue-age monitoring.
- Cancellation propagation.
- Stale data removal.
- Priority for interruption and control events.
- No unbounded memory growth.

Audio must not accumulate behind a delayed model response.

## 18.5 Failure Isolation

An individual provider failure must not crash the session manager or every other session.

Provider connections must be owned and cleaned up independently.

Retries must be bounded and must not replay cancelled user requests incorrectly.

---

# 19. Functional Requirements Summary

| ID | Area | Requirement | Priority |
|---|---|---|---|
| FR-001 | UX | Talk to Sarang entry point | P0 |
| FR-002 | UX | Explicit microphone permission | P0 |
| FR-003 | Media | WebRTC bidirectional audio | P0 |
| FR-004 | Signaling | WebSocket signaling | P0 |
| FR-005 | Session | Session creation and termination | P0 |
| FR-006 | ASR | Streaming English recognition | P0 |
| FR-007 | ASR | Partial transcripts | P0 |
| FR-008 | ASR | Final transcript generation | P0 |
| FR-009 | Speech | VAD and endpointing | P0 |
| FR-010 | LLM | Streaming responses | P0 |
| FR-011 | TTS | Streaming audio generation | P0 |
| FR-012 | Conversation | Multi-turn context | P0 |
| FR-013 | Conversation | Barge-in | P0 |
| FR-014 | Conversation | Generation cancellation | P0 |
| FR-015 | Agent | Fast-path response | P0 |
| FR-016 | Agent | Tool execution | P1 |
| FR-017 | Knowledge | Portfolio document retrieval | P0 |
| FR-018 | UX | Live transcript | P1 |
| FR-019 | UX | Audio visualization | P1 |
| FR-020 | UX | Developer diagnostics | P1 |
| FR-021 | Reliability | Reconnection | P0 |
| FR-022 | Security | Session authentication | P0 |
| FR-023 | Security | API-key protection | P0 |
| FR-024 | Metrics | End-to-end latency | P0 |
| FR-025 | Metrics | ASR WER evaluation | P0 |
| FR-026 | Metrics | TTS RTF measurement | P0 |
| FR-027 | Knowledge | Source-grounded answers | P0 |
| FR-028 | Reliability | Graceful provider failure | P0 |
| FR-029 | Performance | Bounded queues | P0 |
| FR-030 | Privacy | No default recording retention | P0 |

P0 indicates essential for the first usable release. P1 indicates important but not blocking the initial functional release.

---

# 20. Non-Functional Requirements

## 20.1 Performance

- Optimize first-audio latency.
- Minimize per-session memory overhead.
- Keep media processing bounded.
- Avoid synchronous blocking calls on the event loop.
- Support concurrent sessions.
- Measure all major processing stages.
- Keep provider latency separate from network latency.

## 20.2 Reliability

- Correct session cleanup.
- No cross-session data access.
- Bounded retries.
- Connection health monitoring.
- Graceful provider degradation.
- Controlled resource allocation.
- No permanently orphaned background workers.

## 20.3 Scalability

The first release may use a single backend deployment.

The architecture must allow future horizontal scaling.

Future scaling considerations:

- Session-to-worker routing.
- Shared session metadata.
- Worker-level connection limits.
- Distributed rate limiting.
- Redis-backed session coordination.
- Independent ASR/TTS connection pools.
- Media gateway capacity.
- Regional placement.

Do not introduce Kafka, Kubernetes or a microservice fleet without measured requirements.

## 20.4 Maintainability

- Modular provider interfaces.
- Typed events.
- Versioned signaling protocol.
- Reproducible configuration.
- Automated tests.
- Structured logging.
- Provider adapters.
- Documented operational procedures.

## 20.5 Browser Support

Initial browser targets:

- Chrome desktop.
- Edge desktop.
- Firefox desktop.

Mobile browser support should be evaluated as a responsive experience and released once device, permission and playback behavior are validated.

---

# 21. Security, Privacy and Safety

## 21.1 Audio Privacy

The microphone is activated only after visitor consent.

Requirements:

- Explicit microphone permission.
- Visible active-microphone indicator.
- Immediate stop on session termination.
- No default storage of raw recordings.
- No default long-term retention of transcripts.
- Minimal telemetry collection.
- Clear privacy disclosure.

## 21.2 Provider Data Handling

Audio and transcripts may be transmitted to external inference providers.

The portfolio must clearly disclose the relevant processing.

Before production, document:

- Which providers receive data.
- Which regions process the data.
- Provider retention policies.
- Whether provider data may be used for model improvement.
- Applicable contractual and privacy terms.

Do not assume that a provider's default retention policy is suitable for the project without checking it.

## 21.3 Credential Security

- API keys remain backend-only.
- No inference credentials inside frontend code.
- Short-lived session tokens.
- Short-lived TURN credentials.
- Environment-based configuration.
- Secret rotation.
- Rate limiting.
- Origin validation.

## 21.4 Application Security

- Validate WebSocket origins.
- Authenticate session access.
- Prevent session ID enumeration.
- Enforce request size limits.
- Validate every tool request.
- Restrict tools to an allowlist.
- Reject arbitrary shell execution.
- Prevent tool prompt injection.
- Sanitize displayed content.
- Implement per-IP and per-session limits.
- Handle abuse and bot traffic.

## 21.5 Knowledge Security

Only approved public portfolio information should be exposed.

Never make private credentials, personal records or unapproved confidential documents retrievable through the agent.

Retrieved documents must be treated as untrusted information rather than system instructions.

---

# 22. Error Handling and Recovery

The system must define behavior for the following conditions.

| Failure | Expected response |
|---|---|
| Microphone denied | Explain and allow retry |
| No microphone | Offer text fallback or dismiss |
| WebRTC connection fails | Retry or terminate gracefully |
| ICE connection lost | Attempt controlled recovery |
| STT provider unavailable | Display meaningful failure and offer retry |
| LLM timeout | Cancel or retry within defined limits |
| TTS failure | Retry once where safe or show text response |
| Tool timeout | Explain unavailable information |
| Browser disconnect | Clean server-side resources |
| Duplicate messages | Ignore or safely deduplicate |
| Stale audio | Reject by generation ID |
| Queue overload | Apply backpressure and alert |

## 22.1 Text Fallback

Where possible, provide a text input fallback.

The portfolio must remain usable even if microphone permission is unavailable or speech providers are experiencing an outage.

This fallback should not turn the application into a second, separately designed chatbot product.

---

# 23. Observability and Performance Dashboard

## 23.1 Requirement

Sarang must provide a developer diagnostics interface capable of exposing the real-time performance of the system.

This is both an engineering requirement and an important portfolio demonstration.

## 23.2 Metrics

### Browser / WebRTC

- Connection establishment time.
- ICE connection time.
- Selected candidate type.
- RTT.
- Jitter.
- Packet loss.
- Bytes sent.
- Bytes received.
- Playback start.
- Connection state changes.

### ASR

- Audio-to-ASR delivery delay.
- First partial transcript.
- Partial transcript update rate.
- Finalization delay.
- WER.
- Error rate.

### VAD

- Speech onset detection.
- Speech-end detection.
- Endpointing delay.
- False endpoint rate.
- Premature endpoint rate.

### LLM

- Request queue delay.
- Provider TTFT.
- Client-observed TTFT.
- Inter-token latency.
- Tokens per second.
- Total generation duration.
- Token consumption.

### TTS

- First text segment submitted.
- First audio chunk received.
- TTFA.
- Audio generation duration.
- Audio duration.
- RTF.
- Chunk interval.
- Cancellation latency.

### Overall Pipeline

- End-of-speech-to-first-audio.
- Endpoint-to-first-audio.
- Barge-in latency.
- Session failure rate.
- Provider failure rate.
- p50, p95 and p99 latency.

## 23.3 Distributed Tracing

Every turn must have a trace ID.

Every stage must record its own duration using a monotonic clock.

The tracing system must support:

- Session correlation.
- Turn correlation.
- Generation correlation.
- Provider request correlation.
- Error correlation.

Cross-machine timestamps must not be blindly subtracted without accounting for clock offsets.

Use local duration measurement for stage metrics and distributed trace context for correlation.

---

# 24. Product Analytics and Success Metrics

## 24.1 Engagement

- Number of Talk to Sarang activations.
- Microphone permission acceptance rate.
- Successful session establishment rate.
- Average conversation duration.
- Average turns per session.
- Conversation completion rate.
- Text fallback usage.

## 24.2 Answer Quality

- Portfolio answer correctness.
- Retrieval relevance.
- Unsupported claim rate.
- Follow-up coherence.
- Technical explanation quality.
- User abandonment during answers.

Quality should be evaluated using curated test questions and periodic human review.

## 24.3 Operational Success

- Session establishment success.
- Provider success rates.
- Interruption success.
- Audio playback success.
- Error rates.
- Latency percentiles.
- Cost per session.

## 24.4 Product Success Criteria

The initial release will be considered successful when:

1. Visitors can start conversations reliably.
2. Ordinary portfolio questions receive accurate answers.
3. Visitors can ask follow-up questions.
4. Sarang can be interrupted immediately.
5. Audio is generated incrementally.
6. Latency is measurable by stage.
7. The system handles basic network failures.
8. No material cross-session privacy leakage occurs.

---

# 25. Cost and Resource Management

## 25.1 Cost Philosophy

The project should optimize cost without sacrificing its core engineering objectives.

It is not an absolute zero-cost requirement.

The selected providers must be evaluated using actual usage and account-specific pricing.

## 25.2 Cost Measurement

Track:

- ASR audio duration.
- Number of ASR sessions.
- LLM input tokens.
- LLM output tokens.
- Number of model requests.
- TTS characters or generated audio.
- TTS duration.
- Infrastructure runtime.
- TURN relay traffic.
- Storage and retrieval operations.

## 25.3 Cost Controls

- Maximum session duration.
- Idle session timeout.
- Rate limits.
- Maximum conversation context.
- Maximum response length.
- Bounded tool execution.
- Per-session usage limits.
- Provider usage alerts.
- Cost per successful conversation.

## 25.4 Optimization

- Use the fast-path LLM for routine questions.
- Reserve stronger models for complex tasks.
- Avoid unnecessary document retrieval.
- Reuse stable prompt prefixes where supported.
- Cache static portfolio information.
- Avoid repeated provider initialization.
- Limit unnecessary TTS regeneration.

The implementation must not hide latency or usage by relying on unbounded speculative calls.

---

# 26. Deployment Architecture

## 26.1 Initial Deployment

```text
PORTFOLIO FRONTEND
       |
       v
HTTPS WEBSITE
       |
       v
SARANG BACKEND
  FastAPI + aiortc
       |
       +------------------------+
       |             |          |
       v             v          v
    NVIDIA         GROQ      CARTESIA
     ASR            LLM         TTS
       |
       v
POSTGRESQL + PGVECTOR
```

WebRTC media passes through the application gateway.

WebSocket signaling and control messages use a separate connection.

## 26.2 Infrastructure

Initial deployment requires:

- HTTPS frontend.
- Secure backend endpoint.
- WebSocket support.
- WebRTC-compatible network exposure.
- STUN configuration.
- TURN infrastructure.
- Provider API credentials.
- Basic monitoring.
- Logging.
- Rate limiting.

## 26.3 Scaling Strategy

**Stage 1:** Single backend process with independent sessions.

**Stage 2:** Measure concurrent sessions and resource consumption.

**Stage 3:** Separate session orchestration and media processing if bottlenecks justify it.

**Stage 4:** Introduce shared session coordination.

**Stage 5:** Add multiple media gateway workers and regional routing if needed.

Microservices are an architectural option, not an initial requirement.

---

# 27. Delivery Roadmap

## Phase 0 — Technical Validation

Objective: Validate provider connectivity, model access and benchmark methodology.

Tasks:

- Verify NVIDIA Nemotron cloud access.
- Verify supported streaming interface.
- Benchmark 80 ms, 160 ms and 320 ms ASR profiles.
- Test Groq model availability.
- Benchmark LLM TTFT.
- Test Cartesia Sonic WebSocket streaming.
- Verify audio formats.
- Verify provider cancellation.
- Establish baseline network latency.

Deliverable: Provider validation report.

## Sprint 1 — Browser Audio Infrastructure

Objective: Establish bidirectional WebRTC audio transport.

Tasks:

- Microphone capture.
- WebRTC client.
- FastAPI gateway.
- WebSocket signaling.
- SDP exchange.
- Trickle ICE.
- Audio ingress.
- Audio egress.
- Playback.
- Connection metrics.
- Cleanup.

Deliverable: Working bidirectional WebRTC demonstration.

## Sprint 2 — Streaming ASR

Objective: Convert visitor speech into accurate transcripts.

Tasks:

- NVIDIA cloud integration.
- Persistent streaming session.
- Audio normalization.
- ASR aggregation.
- Partial transcript events.
- VAD.
- Endpointing.
- Transcript stabilization.
- WER benchmarks.

Deliverable: Live English speech transcription.

## Sprint 3 — LLM and Conversation Engine

Objective: Make Sarang understand and respond to natural questions.

Tasks:

- Groq provider adapter.
- Model benchmark.
- Prompt architecture.
- Streaming LLM responses.
- Turn manager.
- Conversation state machine.
- Context handling.
- Fast response path.
- Generation cancellation.

Deliverable: Real-time text-based conversational engine.

## Sprint 4 — TTS and Barge-In

Objective: Convert generated responses into streaming speech.

Tasks:

- Cartesia Sonic adapter.
- Persistent WebSocket integration.
- Text segmentation.
- Audio decoding.
- Audio sequencing.
- WebRTC audio egress.
- Browser playback.
- Interruption handling.
- Cancellation propagation.

Deliverable: End-to-end, interruptible voice conversation.

## Sprint 5 — Portfolio Intelligence

Objective: Give Sarang knowledge about the portfolio.

Tasks:

- Document ingestion.
- Portfolio knowledge schema.
- Chunking and embeddings.
- PostgreSQL and pgvector.
- Retrieval implementation.
- Fast-path prompts.
- Tool registry.
- Project-specific tools.
- Evidence-aware response generation.
- Multi-turn question testing.

Deliverable: Portfolio-aware conversational agent.

## Sprint 6 — Performance and Production Hardening

Objective: Optimize the complete pipeline.

Tasks:

- End-to-end latency tracing.
- p50/p95/p99 dashboard.
- Benchmark datasets.
- Barge-in benchmarks.
- Provider failure testing.
- Network degradation testing.
- Concurrency testing.
- Cost instrumentation.
- Security testing.
- Production deployment.
- Final portfolio integration.

Deliverable: Production-oriented Sarang release.

## Future Phase — Advanced Voice Intelligence

Potential future capabilities:

- Custom speaker voice with consent.
- Multilingual recognition.
- Enhanced turn-taking prediction.
- Emotionally expressive speech.
- Adaptive speech rate.
- Advanced speculative generation.
- More sophisticated retrieval.
- Voice activity prediction.
- Automatic interruption recovery.
- Dedicated media gateway infrastructure.

These capabilities are not required for the initial release.

---

# 28. End-to-End Acceptance Tests

## Test 01 — First-Time Visitor

Given a visitor with no previous session,

When they click Talk to Sarang,

Then the microphone permission flow must begin and a new session must be created only after explicit user interaction.

## Test 02 — Natural Question

Given an active session,

When a visitor asks "What is VisionMitra?",

Then Sarang must transcribe the question, retrieve appropriate portfolio knowledge, generate an answer and begin streaming audio.

## Test 03 — Multi-Turn Conversation

Given a completed first answer,

When the visitor asks "What technologies did you use in it?",

Then Sarang must resolve "it" using the previous conversational context.

## Test 04 — Technical Explanation

When a visitor asks for a detailed explanation of a research project,

Then Sarang must retrieve the corresponding documents and distinguish documented results from assumptions.

## Test 05 — Barge-In

Given Sarang is speaking,

When the visitor starts another utterance,

Then playback must stop promptly, the old generation must be invalidated and the system must process the new utterance.

## Test 06 — Slow Tool Execution

Given a complex question requires additional retrieval,

When the tool operation takes longer than the ordinary fast path,

Then Sarang must maintain a responsive conversation interface and must not claim results before they are available.

## Test 07 — Provider Failure

Given an ASR, LLM or TTS provider is unavailable,

When a request reaches that component,

Then the system must handle the failure without crashing other active sessions.

## Test 08 — Session Isolation

Given two visitors are using Sarang simultaneously,

When both transmit speech,

Then their audio, conversation context, generated responses and provider events must remain isolated.

## Test 09 — Network Interruption

Given an active session,

When the network connection is lost,

Then the system must expose the failure, attempt controlled recovery and release resources if recovery fails.

## Test 10 — Latency Measurement

Given a completed voice interaction,

Then the system must record the timestamps and stage durations necessary to calculate:

- End-of-speech-to-first-audio.
- Endpoint-to-first-audio.
- ASR finalization.
- LLM TTFT.
- TTS TTFA.
- Playback scheduling.
- Barge-in cancellation latency.

## Test 11 — Unsupported Question

When a visitor asks something not documented in the portfolio,

Then Sarang must state that the available information is insufficient rather than invent an answer.

## Test 12 — Session Termination

When the visitor ends the session,

Then the microphone, WebRTC peer connection, WebSocket, provider tasks and session resources must be cleaned up.

---

# 29. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| ASR struggles with Indian English | Incorrect transcripts | Build representative evaluation corpus |
| Endpointing cuts off speech | Poor conversation | Adaptive silence thresholds |
| Provider network latency | Missed latency target | Regional benchmarking |
| Slow LLM TTFT | Delayed response | Small-model fast path and prompt optimization |
| TTS startup delay | Delayed speech | Persistent streaming and early text segmentation |
| WebRTC connectivity failure | No conversation | STUN/TURN and reconnection |
| Barge-in false positives | Interrupted answers | Browser AEC and VAD tuning |
| Stale generated audio | Overlapping responses | Generation IDs and sequence validation |
| Tool latency | Slow complex answers | Async execution and concise acknowledgements |
| Incorrect portfolio facts | Loss of credibility | Curated knowledge and evidence tracking |
| Provider outage | System unavailability | Timeouts and graceful degradation |
| Cost spikes | Unexpected operating expense | Session and provider usage limits |

---

# 30. Release Criteria

Sarang can be released as the initial portfolio product once the following conditions are satisfied.

### Functional

- [ ] WebRTC audio works bidirectionally.
- [ ] English streaming ASR works.
- [ ] Natural LLM responses work.
- [ ] Streaming TTS works.
- [ ] Barge-in works.
- [ ] Multi-turn context works.
- [ ] Portfolio retrieval works.
- [ ] Slow-tool handling works.
- [ ] Session cleanup works.

### Quality

- [ ] ASR WER evaluation completed.
- [ ] Portfolio question set validated.
- [ ] Incorrect factual response rate reviewed.
- [ ] Voice quality reviewed.
- [ ] Interruption behavior reviewed.
- [ ] Error recovery tested.

### Performance

- [ ] End-to-end latency instrumented.
- [ ] p50/p95/p99 measurements available.
- [ ] Component targets benchmarked.
- [ ] Audio queue behavior validated.
- [ ] Concurrency tests completed.
- [ ] The sub-500 ms objective evaluated under declared conditions.

### Security

- [ ] Provider credentials secured.
- [ ] WebSocket authentication implemented.
- [ ] Session isolation tested.
- [ ] Rate limits enabled.
- [ ] Microphone privacy behavior verified.
- [ ] No unintended raw audio retention.

### Deployment

- [ ] HTTPS configured.
- [ ] WebSocket endpoint available.
- [ ] TURN connectivity tested.
- [ ] Provider configuration verified.
- [ ] Health checks enabled.
- [ ] Basic monitoring enabled.
- [ ] Recovery procedures documented.

---

# 31. Definition of Product Success

Sarang is successful when visitors can have a natural technical conversation with the portfolio rather than simply interacting with a voice-enabled chatbot.

The engineering objectives are:

1. A responsive and interruptible conversational experience.
2. Accurate English speech recognition.
3. Low-latency streaming LLM inference.
4. Fast, natural streaming TTS.
5. Grounded portfolio knowledge.
6. Reliable real-time audio transport.
7. Correct concurrency and cancellation.
8. Measurable end-to-end performance.
9. Graceful handling of provider and network failures.
10. Secure, privacy-conscious operation.

The ultimate performance ambition is sub-500 ms end-of-speech-to-first-audible-response latency under controlled, documented conditions.

This is an optimization target that must be proven through actual measurements.

---

# 32. Final Product Definition

**Sarang is a custom, event-driven, browser-based conversational AI system using a cascaded streaming speech architecture.**

It combines:

- WebRTC for real-time bidirectional audio.
- NVIDIA Nemotron for streaming English speech recognition.
- Custom VAD and endpointing.
- Groq for low-latency streaming language generation.
- Portfolio-focused retrieval and controlled tool execution.
- Cartesia Sonic for incremental speech synthesis.
- Custom audio orchestration and playback.
- Generation-aware cancellation and barge-in.
- Distributed tracing and performance engineering.

The product is designed around one fundamental requirement:

**The visitor should be able to talk naturally, hear a fast and accurate response, interrupt whenever they want, and explore the portfolio without needing to understand the technology operating behind the conversation.**/