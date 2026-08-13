| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-af1c43ae52b416f1fb7e0befc199ddfc1f62e0b3.md) | 464 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-af1c43ae52b416f1fb7e0befc199ddfc1f62e0b3.md) | 7,349 |  14,365,133 |  1,512 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-af1c43ae52b416f1fb7e0befc199ddfc1f62e0b3.md) | 4,172 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-af1c43ae52b416f1fb7e0befc199ddfc1f62e0b3.md) | 664 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-af1c43ae52b416f1fb7e0befc199ddfc1f62e0b3.md) | 196 |  112,210 |  196 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-af1c43ae52b416f1fb7e0befc199ddfc1f62e0b3.md) | 237 |  592,827 |  195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-af1c43ae52b416f1fb7e0befc199ddfc1f62e0b3.md) | 2,025 |  1,979,971 |  532 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/af1c43ae52b416f1fb7e0befc199ddfc1f62e0b3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31717291106)
