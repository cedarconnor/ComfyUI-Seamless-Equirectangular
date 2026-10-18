# Recorded validation: October 18, 2026

Compact evidence from the local ComfyUI MCP investigation:

- [Job IDs](all_jobs.json) and [completion status messages](job_status_archive.json): 34 completed jobs. Three earlier guide comparisons are excluded as stock controls because they used the old global RoPE implementation.
- [Cache isolation](cache_isolation_runtime.json): 65 stock frames reproduced exactly after running an enabled spherical branch, while cached model nodes were reused and the sampler executed afresh.
- [UI/API comparison](ui_runtime_comparison.json): both delivered LTX UI workflows reproduce all 65 API reference frames exactly.
- [Longer-clip metrics](long_metrics_summary.json): mean and maximum boundary-jump ratios for 65-frame LTX, 1536x768 LTX and 56-frame H3 tests.
- [Worst-edge review](worst_edge_review.json): a high boundary gradient can be a legitimate window edge crossing the join.

For RGB PNG values normalized to [0, 1], the metric uses `d = abs(image - roll(image, 1, width_axis)).mean(height_and_color_axes)` and `ratio = d[0] / max(median(d), 1e-8)`. The denominator includes all columns, including the wrap boundary; the report's shorthand "interior difference" refers to this median reference. Lower scores alone do not establish visual continuity, stable geometry or correct poles. See the [report](../../SEAM_DIAGNOSIS_2026-10-18.md) for settings and interpretation.

Raw frames, latents, full execution histories, projection review media and the local delivery ZIP are not bundled in Git. The report's `output/seam_diagnosis_20261018` paths refer to that local diagnostic archive. Included [API graphs](../../../example_workflows/seam_diagnosis_20261018/README.md) and [UI presets](../../../example_workflows/ltx25/README.md) provide the selected starting points; reproduction requires the named models and suitable input media.
