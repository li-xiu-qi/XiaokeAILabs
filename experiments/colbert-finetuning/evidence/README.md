# Evidence notes

`results_residual_compression.json` is retained as historical output from a synthetic demonstration. The previous script hard-coded its reported 0.982 recall; that number was not measured. Brute-force timing was extrapolated from a 100-document sample, and the pruning path scored at most 100 candidates. The file therefore does not establish retrieval recall or end-to-end speedup. The current script marks recall as unmeasured and refuses to overwrite an existing result file.

The other files in this directory document the separate fine-tuning comparison described in the [experiment README](../README.en.md).
