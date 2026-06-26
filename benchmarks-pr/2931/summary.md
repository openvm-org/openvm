| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-d87d7dda28d81b3b08277d721cb3ec57c3ab8271.md) | 1,023 |  4,000,051 |  388 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-d87d7dda28d81b3b08277d721cb3ec57c3ab8271.md) | 15,707 |  14,365,133 |  3,017 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-d87d7dda28d81b3b08277d721cb3ec57c3ab8271.md) | 8,265 |  11,167,961 |  1,012 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-d87d7dda28d81b3b08277d721cb3ec57c3ab8271.md) | 1,166 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-d87d7dda28d81b3b08277d721cb3ec57c3ab8271.md) | 443 |  112,210 |  278 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-d87d7dda28d81b3b08277d721cb3ec57c3ab8271.md) | 586 |  592,827 |  291 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-d87d7dda28d81b3b08277d721cb3ec57c3ab8271.md) | 3,852 |  1,979,971 |  855 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d87d7dda28d81b3b08277d721cb3ec57c3ab8271

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28266914016)
