# Research — Rogue Nodes in Multi-Access Edge Computing

## The paper

"Lightweight Trust-Based Detection of Rogue Nodes in MEC." First author. The
venue is AEUE, a Q2 journal. The status is **submitted**: not accepted, not
published, peer review not complete.

## The problem

Multi-access edge computing moves compute next to the radio access network, so
workloads run on nodes the operator does not physically control. A compromised
or malicious node inside that topology can drop, delay or alter traffic while
still appearing to be part of the network. Detecting it has to be cheap, because
an edge node is a constrained device, not a data-centre server.

## The mechanism

The framework is SDN-based. A controller observes the topology and computes a
trust signal per node rather than relying on a heavyweight intrusion-detection
model on every device. Behaviour that departs from the node's expected role
lowers its trust score, and a node that falls far enough is treated as rogue and
isolated from the forwarding path.

Because the controller already sees the network view, the detection logic can
live there. That is what keeps the per-node cost low: the endpoints do not need
to run the classifier.

## Validation

The framework was implemented and validated with Python, Mininet and Ryu-based
simulation. The reported results are a detection accuracy of ninety-eight
percent, CPU usage under three percent and added delay under fifty milliseconds.

## Honest limits

The results come from simulation, not a production 5G deployment. A simulated
topology is cleaner than a real one: it does not have to survive misconfigured
hardware, vendor-specific control planes or adversarial traffic at scale. The
paper is submitted, so the claims are pre-review.

## Why the topic

Edge security is a latency problem as much as a security problem. A detector
that adds fifty milliseconds to a forwarding path has already broken the reason
the edge exists. Keeping the mechanism lightweight was the design constraint,
not an afterthought.
