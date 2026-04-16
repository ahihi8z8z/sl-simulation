# CLAUDE.md

## Project

Serverless computing simulator with RL-based autoscaling. SimPy-based discrete event simulation.

## Environment

- Python env: `conda activate sl-venv`
- Run tests: `python -m pytest tests/ -q`
- Run simulation: `python -m serverless_sim.runtime.cli simulate --sim-config <config.json>`
- Train RL: `python -m serverless_sim.runtime.cli train --sim-config <config> --rl-config <rl_config>`

## Code change workflow

1. Ask before making changes — don't assume intent from questions
2. Plan first: flow diagram, changes per file, risks
3. No backward compatibility — delete replaced code, don't keep dead code
4. Ask before committing
5. Always run tests

## Design principles

- Modules should be independent and configurable separately
- Prefer simple attribute tracking over complex class hierarchies
- SimPy processes: `env.process()` only schedules, doesn't execute until `env.run()` — be aware of this when counting/checking state
- experimental/ is gitignored
