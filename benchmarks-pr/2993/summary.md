| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-90549d1414678a56490dcff24c341f7fee522af4.md) | 409 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-90549d1414678a56490dcff24c341f7fee522af4.md) | 8,584 |  14,365,133 |  1,511 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-90549d1414678a56490dcff24c341f7fee522af4.md) | 4,178 |  11,167,961 |  516 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-90549d1414678a56490dcff24c341f7fee522af4.md) | 568 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-90549d1414678a56490dcff24c341f7fee522af4.md) | 222 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-90549d1414678a56490dcff24c341f7fee522af4.md) | 298 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-90549d1414678a56490dcff24c341f7fee522af4.md) | 1,943 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/90549d1414678a56490dcff24c341f7fee522af4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29512856651)
