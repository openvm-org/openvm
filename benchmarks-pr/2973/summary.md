| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/fibonacci-ab8c955ef5761065d66d0467a5abaf4ce5fb7467.md) | 442 |  4,000,051 |  226 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/keccak-ab8c955ef5761065d66d0467a5abaf4ce5fb7467.md) | 8,379 |  14,365,133 |  1,520 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/sha2_bench-ab8c955ef5761065d66d0467a5abaf4ce5fb7467.md) | 4,159 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/regex-ab8c955ef5761065d66d0467a5abaf4ce5fb7467.md) | 503 |  4,090,656 |  193 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/ecrecover-ab8c955ef5761065d66d0467a5abaf4ce5fb7467.md) | 220 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/pairing-ab8c955ef5761065d66d0467a5abaf4ce5fb7467.md) | 274 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/kitchen_sink-ab8c955ef5761065d66d0467a5abaf4ce5fb7467.md) | 1,997 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ab8c955ef5761065d66d0467a5abaf4ce5fb7467

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29399852269)
