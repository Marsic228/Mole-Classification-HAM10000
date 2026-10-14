# Research Archive

This directory preserves historical Python scripts that are no longer
part of the active training and evaluation workflow.

diagnostics/ - Earlier learning exercises, checks and diagnostics.

Files are preserved for research reproducibility, not deleted.

When running an archived script, launch it from the repository root
as a Python module, for example:

python -m archive.diagnostics.model_forward_check

Some historical scripts require their original dependencies,
dataset files and Python environment.

Do not move shared utilities or historical training implementations
without checking their imports and evaluation dependencies.
