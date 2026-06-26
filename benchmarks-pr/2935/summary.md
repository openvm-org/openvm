| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2935/fibonacci-980a3df1485d881319749b9443aad8401b6c964f.md) | 1,037 |  4,000,051 |  394 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2935/keccak-980a3df1485d881319749b9443aad8401b6c964f.md) | 16,194 |  14,365,133 |  3,017 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2935/sha2_bench-980a3df1485d881319749b9443aad8401b6c964f.md) | 8,311 |  11,167,961 |  1,004 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2935/regex-980a3df1485d881319749b9443aad8401b6c964f.md) | 1,197 |  4,090,656 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2935/ecrecover-980a3df1485d881319749b9443aad8401b6c964f.md) | 445 |  112,210 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2935/pairing-980a3df1485d881319749b9443aad8401b6c964f.md) | 595 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2935/kitchen_sink-980a3df1485d881319749b9443aad8401b6c964f.md) | 3,888 |  1,979,971 |  861 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/980a3df1485d881319749b9443aad8401b6c964f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28247768638)
