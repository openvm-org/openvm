| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/fibonacci-18b7c7b658b11d35d86a5593258c5d723f8585ce.md) | 1,011 |  4,000,051 |  389 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/keccak-18b7c7b658b11d35d86a5593258c5d723f8585ce.md) | 15,215 |  14,365,133 |  2,987 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/sha2_bench-18b7c7b658b11d35d86a5593258c5d723f8585ce.md) | 7,707 |  11,167,961 |  984 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/regex-18b7c7b658b11d35d86a5593258c5d723f8585ce.md) | 1,155 |  4,090,656 |  350 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/ecrecover-18b7c7b658b11d35d86a5593258c5d723f8585ce.md) | 438 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/pairing-18b7c7b658b11d35d86a5593258c5d723f8585ce.md) | 564 |  592,827 |  293 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/kitchen_sink-18b7c7b658b11d35d86a5593258c5d723f8585ce.md) | 3,748 |  1,979,971 |  855 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/18b7c7b658b11d35d86a5593258c5d723f8585ce

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28098232197)
