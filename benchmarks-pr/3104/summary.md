| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-d231520aa612ee4fef0369074242268e7fb1b84f.md) | 887 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-d231520aa612ee4fef0369074242268e7fb1b84f.md) | 8,645 |  14,365,133 |  1,514 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-d231520aa612ee4fef0369074242268e7fb1b84f.md) | 4,224 |  11,167,961 |  516 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-d231520aa612ee4fef0369074242268e7fb1b84f.md) | 738 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-d231520aa612ee4fef0369074242268e7fb1b84f.md) | 497 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-d231520aa612ee4fef0369074242268e7fb1b84f.md) | 479 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-d231520aa612ee4fef0369074242268e7fb1b84f.md) | 2,350 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d231520aa612ee4fef0369074242268e7fb1b84f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30957113928)
