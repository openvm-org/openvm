| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/fibonacci-53752bdfbf573ff4cf0c09e275803ee0d8bdc52a.md) | 478 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/keccak-53752bdfbf573ff4cf0c09e275803ee0d8bdc52a.md) | 7,486 |  14,365,133 |  1,531 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/sha2_bench-53752bdfbf573ff4cf0c09e275803ee0d8bdc52a.md) | 4,261 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/regex-53752bdfbf573ff4cf0c09e275803ee0d8bdc52a.md) | 748 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/ecrecover-53752bdfbf573ff4cf0c09e275803ee0d8bdc52a.md) | 214 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/pairing-53752bdfbf573ff4cf0c09e275803ee0d8bdc52a.md) | 260 |  592,827 |  193 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/kitchen_sink-53752bdfbf573ff4cf0c09e275803ee0d8bdc52a.md) | 2,234 |  1,979,971 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/53752bdfbf573ff4cf0c09e275803ee0d8bdc52a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/37040114700)
