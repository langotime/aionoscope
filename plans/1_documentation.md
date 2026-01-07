# Plan: Documentation Strategy for ToyTS

## Goal
Establish a documentation standard that ensures ToyTS is accessible to human researchers while being unambiguous and easy to navigate for AI coding agents.

## 1. Documentation Principles

### 1.1. For AI Agents (Precision & Context)
AI agents rely on explicit contracts and context.
- **Type Hints**: strict usage of Python type hints (PEP 484) in all public APIs.
- **Tensor Shapes**: Every tensor argument and return value must have a shape comment (e.g., `# [B, C, L]`).
- **Self-Contained Examples**: `toyts/examples/` should contain runnable scripts that demonstrate correct usage patterns. Agents can read these to understand how to compose modules.
- **Docstrings**: Google-style docstrings for all classes and `forward` methods.

### 1.2. For Humans (Concepts & Onboarding)
- **Conceptual Overview**: Explain the *Process vs. View* philosophy clearly (as done in the README).
- **Visuals**: Use ASCII diagrams or Mermaid charts in Markdown to explain the data flow.
- **Progressive Disclosure**: Start with simple single-view generation, then move to multi-view SSL.

## 2. Implementation Plan

### Phase 1: In-Code Documentation (Definition of Done for Code)
Every PR/Commit introducing a new Process or View must include:
1.  **Class Docstring**: What does this module simulate? What are the parameters?
2.  **Forward Docstring**:
    ```python
    def forward(self, z: LatentState) -> Observation:
        """
        Args:
            z: Latent state with components [B, K, L]
        Returns:
            Observation with x [B, C, L_out]
        """
    ```
3.  **Shape Annotations**: Inline comments for complex tensor operations (einsum, reshape).

### Phase 2: Runnable Examples
Create a directory `toyts/examples/` with minimal scripts:
- `01_simple_pulse.py`: Basic generation.
- `02_multiview_ssl.py`: Generating pairs for contrastive learning.
- `03_server_metrics.py`: Demonstrating the `TrendSeasonAnomaly` process.

These scripts serve as the primary "How-To" for both humans and agents.

### Phase 3: Context Aggregation (for Agents)
Create a `toyts/CONTEXT.md` (or similar) that concatenates the public interfaces of `core`, `processes`, and `views`. This allows an agent to quickly ingest the entire library surface area in one read.

## 3. Style Guide

*   **Terminology**:
    *   Use `B` for Batch size.
    *   Use `C` for Channels (leads/sensors).
    *   Use `L` for Length (time steps).
    *   Use `K` for Latent components.
*   **Config**: Prefer explicit `__init__` arguments over giant config dictionaries for individual modules.
*   **Reproducibility**: Document how `rng` (torch.Generator) is passed down.

---
