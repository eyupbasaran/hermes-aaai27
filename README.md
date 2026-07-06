# HERMES AAAI-27

HERMES is an event-driven resource scheduler for multi-source molecular active search.

The first milestone is M0: a QeMFi wall-clock replay harness that evaluates whether explicit `(candidate, source, backend)` scheduling improves high-fidelity discovery utility under asynchronous heterogeneous resource constraints.

## Current Status

This repository is scaffolded from the handoff package in `../hermes_agent_handoff`.

Implementation has not started yet. The next milestone is to fill in the core dataclasses, simulator API, QeMFi loader, and tests described in the handoff notes.

## Non-Negotiable Design Rule

Every launched job must be represented as:

```python
(candidate_id, source_id, backend_id)
```

The source determines the scientific observation. The backend determines runtime, cost, capacity, startup latency, queue behavior, and batching.

## Intended Commands

```bash
pip install -e .
pytest -q
python -m hermes.experiments.run_one --config configs/experiment/qemfi_m0_debug.yaml
python -m hermes.experiments.run_sweep --config configs/experiment/qemfi_m0_main.yaml
python -m hermes.experiments.aggregate --results results/qemfi_m0
python -m hermes.experiments.plot --results results/qemfi_m0
```

