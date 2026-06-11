| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/fibonacci-583c1ab400fd0ab34b2f0b10ace19ab3343638a8.md) | 1,682 |  4,000,051 |  529 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/keccak-583c1ab400fd0ab34b2f0b10ace19ab3343638a8.md) | 16,729 |  14,365,133 |  3,102 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/sha2_bench-583c1ab400fd0ab34b2f0b10ace19ab3343638a8.md) | 10,384 |  11,167,961 |  1,929 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/regex-583c1ab400fd0ab34b2f0b10ace19ab3343638a8.md) | 1,532 |  4,090,656 |  429 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/ecrecover-583c1ab400fd0ab34b2f0b10ace19ab3343638a8.md) | 479 |  112,210 |  309 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/pairing-583c1ab400fd0ab34b2f0b10ace19ab3343638a8.md) | 620 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/kitchen_sink-583c1ab400fd0ab34b2f0b10ace19ab3343638a8.md) | 3,955 |  1,979,971 |  858 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/583c1ab400fd0ab34b2f0b10ace19ab3343638a8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27354791129)
