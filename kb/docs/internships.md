# Internships

Three internships, each with a different constraint: a production workflow, a
data-residency boundary, and a research simulation.

## EPIC Technology Research — Agentic AI Engineer Intern

May to July 2026, Pune. Built the Municipal Asset Intelligence Copilot: an
agentic system that answers multi-step questions across millions of municipal
asset records. The work covered intent-aware query routing, hybrid retrieval
over PostgreSQL and pgvector, and on-premises LLM orchestration. The on-prem
requirement came from data residency, so the orchestration had to treat the
model as a local resource with its own queueing and failure behaviour.

There are no published accuracy or latency figures for this internship. The
scale claim — millions of records — is documented; a benchmark number is not.

## ARAI — Python Developer Intern

December 2025 to February 2026, Pune. Automotive Research Association of India.
Built and deployed a production Python automation pipeline on the ZEISS INSPECT
API for industrial photogrammetry. The pipeline automated marker association,
validation and quality checks that were previously manual.

The measured result: a workflow that took ten hours dropped to about ten
seconds, a processing-time reduction of 99.94 percent. That number came from the
résumé. The deployment was single-team inside a research association, not a
public product.

## COEP Technological University — Research Intern, Network Security and 5G

May to July 2025, Pune. Developed a lightweight SDN-based rogue-node detection
framework for 5G multi-access edge computing, validated with Python, Mininet and
Ryu simulation. The work produced the AEUE paper submission described elsewhere
in this portfolio. Results are simulation-based: ninety-eight percent detection
accuracy, CPU usage under three percent and added delay under fifty
milliseconds.

## What the three have in common

Each one forced a real constraint rather than a demo constraint. ARAI had a live
workflow that had to keep working. EPIC had records that were not allowed to
leave the premises. COEP had to justify every millisecond it added to a
forwarding path. Those constraints are the interesting part of the engineering.
