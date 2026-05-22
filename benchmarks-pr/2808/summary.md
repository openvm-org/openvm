| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/fibonacci-b09573e660eb38d9d73e8ba40016bf49fb969e88.md) | 1,586 |  4,000,051 |  436 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/keccak-b09573e660eb38d9d73e8ba40016bf49fb969e88.md) | 13,883 |  14,365,133 |  2,370 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/sha2_bench-b09573e660eb38d9d73e8ba40016bf49fb969e88.md) | 9,354 |  11,167,961 |  1,431 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/regex-b09573e660eb38d9d73e8ba40016bf49fb969e88.md) | 1,479 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/ecrecover-b09573e660eb38d9d73e8ba40016bf49fb969e88.md) | 477 |  112,210 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/pairing-b09573e660eb38d9d73e8ba40016bf49fb969e88.md) | 595 |  592,827 |  256 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/kitchen_sink-b09573e660eb38d9d73e8ba40016bf49fb969e88.md) | 1,831 |  1,979,971 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b09573e660eb38d9d73e8ba40016bf49fb969e88

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26291408302)
