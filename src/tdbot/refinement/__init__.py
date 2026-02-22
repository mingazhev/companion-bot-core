"""Refinement worker package.

Provides async prompt refinement via a non-interactive LLM call:
    - schemas     — job envelope, delta, and result types
    - client      — refine_prompt() using ChatAPIClient
    - validator   — policy checks on refinement output
    - worker      — queue consumer that applies validated deltas
    - scheduler   — cadence-based job scheduling helpers
"""
