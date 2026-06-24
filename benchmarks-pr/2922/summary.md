| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/fibonacci-6dcd7ade21b6365de420ce9bd181e865e0879734.md) | 1,041 |  4,000,051 |  398 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/keccak-6dcd7ade21b6365de420ce9bd181e865e0879734.md) | 15,355 |  14,365,133 |  3,029 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/sha2_bench-6dcd7ade21b6365de420ce9bd181e865e0879734.md) | 7,885 |  11,167,961 |  1,006 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/regex-6dcd7ade21b6365de420ce9bd181e865e0879734.md) | 1,153 |  4,090,656 |  348 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/ecrecover-6dcd7ade21b6365de420ce9bd181e865e0879734.md) | 437 |  112,210 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/pairing-6dcd7ade21b6365de420ce9bd181e865e0879734.md) | 564 |  592,827 |  301 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/kitchen_sink-6dcd7ade21b6365de420ce9bd181e865e0879734.md) | 3,795 |  1,979,971 |  858 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6dcd7ade21b6365de420ce9bd181e865e0879734

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28084498164)
