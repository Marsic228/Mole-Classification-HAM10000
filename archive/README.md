# Research Archive

This directory preserves historical experiment and diagnostic Python scripts that are no longer part of active training. Keep the originals and their reports for research traceability, even when a method was rejected or its evaluation later proved invalid.

- `diagnostics/` — earlier exercises, data checks, and limited-pipeline validation.
- `legacy_experiments/` — historical ResNet18, EfficientNet-B0, MobileNetV3 and older ConvNeXt code, including standalone train/evaluation variants.

Run Python modules **from the repository root**, for example:

```powershell
.\venv\Scripts\python.exe -m archive.diagnostics.model_forward_check
```

Some archived scripts have module-level execution, legacy relative paths or imports, and depend on old model checkpoints and distinct environments. A successful Python syntax or import check is **not** a promise that re-running the script reproduces its original results. Inspect the script and experiment report first; never run an archived training/evaluation command merely to test imports if it may overwrite historical files.

Shared code now lives in `core/`, `evaluation/` and `preprocessing/`, **not at the repository root**. Historical checkpoints from the original ConvNeXt and PanDerm environments can have different numerical fold assignments despite identical seed/configuration; preserve and verify the actual validation image and lesion IDs.

See `Diploma_Experiment_History.md` (project research record) and `reports/README.md` for detailed decisions and metric evidence. Reports with `reference_check_passed: false` are retained as error evidence and must not be used as valid metrics.
