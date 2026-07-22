| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/fibonacci-38825dff33646dd83955b6e62de5712188b76f1a.md) | 475 |  4,000,051 |  243 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/keccak-38825dff33646dd83955b6e62de5712188b76f1a.md) | 7,317 |  14,365,133 |  1,555 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/sha2_bench-38825dff33646dd83955b6e62de5712188b76f1a.md) | 4,622 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/regex-38825dff33646dd83955b6e62de5712188b76f1a.md) | 673 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/ecrecover-38825dff33646dd83955b6e62de5712188b76f1a.md) | 228 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/pairing-38825dff33646dd83955b6e62de5712188b76f1a.md) | 325 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/kitchen_sink-38825dff33646dd83955b6e62de5712188b76f1a.md) | 2,674 |  1,979,971 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/38825dff33646dd83955b6e62de5712188b76f1a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29899349115)
