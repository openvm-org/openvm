| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/fibonacci-171466b053c81da42c3f47ff13be902918b2cb11.md) | 474 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/keccak-171466b053c81da42c3f47ff13be902918b2cb11.md) | 7,356 |  14,365,133 |  1,531 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/sha2_bench-171466b053c81da42c3f47ff13be902918b2cb11.md) | 4,139 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/regex-171466b053c81da42c3f47ff13be902918b2cb11.md) | 663 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/ecrecover-171466b053c81da42c3f47ff13be902918b2cb11.md) | 223 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/pairing-171466b053c81da42c3f47ff13be902918b2cb11.md) | 238 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/kitchen_sink-171466b053c81da42c3f47ff13be902918b2cb11.md) | 2,041 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/171466b053c81da42c3f47ff13be902918b2cb11

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30950289745)
