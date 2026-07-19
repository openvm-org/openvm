| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/fibonacci-2b55bfa4a01e5e927733f262be139a3a1633aa38.md) | 416 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/keccak-2b55bfa4a01e5e927733f262be139a3a1633aa38.md) | 8,659 |  14,365,133 |  1,526 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/sha2_bench-2b55bfa4a01e5e927733f262be139a3a1633aa38.md) | 4,223 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/regex-2b55bfa4a01e5e927733f262be139a3a1633aa38.md) | 560 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/ecrecover-2b55bfa4a01e5e927733f262be139a3a1633aa38.md) | 227 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/pairing-2b55bfa4a01e5e927733f262be139a3a1633aa38.md) | 292 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/kitchen_sink-2b55bfa4a01e5e927733f262be139a3a1633aa38.md) | 2,016 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2b55bfa4a01e5e927733f262be139a3a1633aa38

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29684197140)
