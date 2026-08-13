| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-6728448e1f272c32a9aa23e6c32b71b5f58925a4.md) | 448 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-6728448e1f272c32a9aa23e6c32b71b5f58925a4.md) | 7,227 |  14,365,133 |  1,594 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-6728448e1f272c32a9aa23e6c32b71b5f58925a4.md) | 4,140 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-6728448e1f272c32a9aa23e6c32b71b5f58925a4.md) | 719 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-6728448e1f272c32a9aa23e6c32b71b5f58925a4.md) | 210 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-6728448e1f272c32a9aa23e6c32b71b5f58925a4.md) | 235 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-6728448e1f272c32a9aa23e6c32b71b5f58925a4.md) | 2,169 |  1,979,971 |  456 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6728448e1f272c32a9aa23e6c32b71b5f58925a4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31637846732)
