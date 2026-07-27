| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/fibonacci-d76e04e28eb43da83dc7b31f9a37d11cabd7be47.md) | 471 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/keccak-d76e04e28eb43da83dc7b31f9a37d11cabd7be47.md) | 7,275 |  14,365,133 |  1,520 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/sha2_bench-d76e04e28eb43da83dc7b31f9a37d11cabd7be47.md) | 4,733 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/regex-d76e04e28eb43da83dc7b31f9a37d11cabd7be47.md) | 671 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/ecrecover-d76e04e28eb43da83dc7b31f9a37d11cabd7be47.md) | 223 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/pairing-d76e04e28eb43da83dc7b31f9a37d11cabd7be47.md) | 267 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/kitchen_sink-d76e04e28eb43da83dc7b31f9a37d11cabd7be47.md) | 2,728 |  1,979,971 |  475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d76e04e28eb43da83dc7b31f9a37d11cabd7be47

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30311347938)
