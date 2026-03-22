# Changelog

## 2.3.0 - 2026-03-22
1. Added end-to-end pipeline orchestrator for config-driven capture, training, and splat cleanup.
2. Added headless multi-view capture script for GUI-free batch rendering.
3. Added asset library batch capture with introspection, filtering, adaptive analysis, and parallel execution.
4. Added proxy-hull splat cleanup with safety controls and CI smoke coverage.
5. Added config schema validation (capture and library job configs) with typed dataclasses.
6. Added 700+ unit tests across 8 test suites with ruff linting and mkdocs strict CI.
7. Hardened config file loading with clear error messages for missing files and parse failures.
8. Added documentation for headless capture, library batch capture, and automation pipeline guides.

## 2.2.2 - 2026-02-21
1. Fixed training startup to use preflight-normalized paths and robust extra-arg parsing.
2. Hardened training stop behavior to terminate full process trees cross-platform.
3. Strengthened checkpoint resume validation so missing/corrupt outputs force a fresh capture.
4. Added safe output directory creation handling to avoid uncaught path/permission crashes.
5. Added release gates for tag/version consistency plus Python and Blender smoke verification before publish.

## 2.2.1 - 2026-02-07
1. Hardened capture pipeline with safer checkpoint/resume handling and output validation.
2. Added coverage validation using camera visibility stats with large-mesh safeguards.
3. Improved Windows compatibility for training backend discovery and path length warnings.
4. Refreshed documentation and added release checklist and packaging guidance.
