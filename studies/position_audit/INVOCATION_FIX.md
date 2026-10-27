# Launch 001: evidence path correction

The first reconstruction launch stopped before executing the upstream script or measuring performance. The runner changes into a temporary filesystem stage, so its relative --evidence argument no longer resolved. Raw traceback and initial provenance are preserved under evidence/launch_001. No model or GPU was involved.

Correction: invoke the exact same frozen runner with an absolute --evidence path and a new --out directory. No source, math, data, bootstrap count or success criterion changes. Freeze3da8543 remains intact. This is an invocation fix, not a result-informed scientific amendment. Future reproduction commands must use absolute output/evidence paths.
