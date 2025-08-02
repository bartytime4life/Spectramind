"""
SpectraMind V50 – Execution Flow Map (Ultimate Version)
--------------------------------------------------------
Defines ordered runtime stages for diagnostics, documentation, and profiling.
Integrates with:
- generate_html_report.py (system overview section)
- cli_diagnose.py (prints runtime DAG overview)
- v50_debug_log.md (records errors with stage context)
- error_humanizer.py (includes origin stage in trace headers)
- selftest.py (verifies presence of required pipeline stages)
"""

from typing import List, Dict

class ExecutionFlow:
    def __init__(self):
        self.stages = [
            "load_raw_frames",
            "calibrate_detector",
            "extract_lightcurves",
            "extract_features",
            "encode_fgs",
            "encode_airs",
            "combine_latents",
            "predict_mu_sigma",
            "apply_symbolic_constraints",
            "compute_loss",
            "backpropagation"
        ]

        self._descriptions = {
            "load_raw_frames": "Read .parquet detector files (FGS1, AIRS)",
            "calibrate_detector": "Apply ADC, dark, flat, linear corrections",
            "extract_lightcurves": "CDS + photometry + spectral trace",
            "extract_features": "Downsample FGS1, normalize, attach metadata",
            "encode_fgs": "Mamba SSM encodes white-light FGS1",
            "encode_airs": "GNN encodes spectral AIRS input",
            "combine_latents": "Concatenate AIRS, FGS, and planet metadata",
            "predict_mu_sigma": "Final decoder outputs μ and σ",
            "apply_symbolic_constraints": "Apply symbolic loss and photonic logic",
            "compute_loss": "Total = GLL + symbolic + optional regularizers",
            "backpropagation": "Backward pass and optimizer step"
        }

    def describe(self) -> List[Dict[str, str]]:
        return [
            {"stage": s, "description": self.describe_stage(s)} for s in self.stages
        ]

    def describe_stage(self, stage: str) -> str:
        return self._descriptions.get(stage, "(no description)")

    def print_summary(self):
        print("\n🧠 SpectraMind V50 – Execution Flow Stages")
        for i, s in enumerate(self.stages):
            desc = self._descriptions.get(s, "")
            print(f"[{i+1:02}] {s:25s} - {desc}")


if __name__ == "__main__":
    flow = ExecutionFlow()
    flow.print_summary()