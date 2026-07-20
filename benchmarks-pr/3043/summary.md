| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/fibonacci-8e4de9ab7ab425c1203b4d566545003bf13a7097.md) | 411 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/keccak-8e4de9ab7ab425c1203b4d566545003bf13a7097.md) | 8,704 |  14,365,133 |  1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/sha2_bench-8e4de9ab7ab425c1203b4d566545003bf13a7097.md) | 4,235 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/regex-8e4de9ab7ab425c1203b4d566545003bf13a7097.md) | 577 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/ecrecover-8e4de9ab7ab425c1203b4d566545003bf13a7097.md) | 220 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/pairing-8e4de9ab7ab425c1203b4d566545003bf13a7097.md) | 292 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/kitchen_sink-8e4de9ab7ab425c1203b4d566545003bf13a7097.md) | 1,923 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8e4de9ab7ab425c1203b4d566545003bf13a7097

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29765272519)
