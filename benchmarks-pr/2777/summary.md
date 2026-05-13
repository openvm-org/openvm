| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/fibonacci-d6a17790a9319234b0d8067435234b83c4674a93.md) | 1,849 |  4,000,051 |  456 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/keccak-d6a17790a9319234b0d8067435234b83c4674a93.md) | 13,987 |  14,365,133 |  2,227 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/sha2_bench-d6a17790a9319234b0d8067435234b83c4674a93.md) | 8,407 |  11,167,961 |  933 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/regex-d6a17790a9319234b0d8067435234b83c4674a93.md) | 1,615 |  4,090,656 |  386 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/ecrecover-d6a17790a9319234b0d8067435234b83c4674a93.md) | 638 |  112,210 |  291 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/pairing-d6a17790a9319234b0d8067435234b83c4674a93.md) | 749 |  592,827 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/kitchen_sink-d6a17790a9319234b0d8067435234b83c4674a93.md) | 2,018 |  1,979,971 |  429 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d6a17790a9319234b0d8067435234b83c4674a93

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25829174242)
