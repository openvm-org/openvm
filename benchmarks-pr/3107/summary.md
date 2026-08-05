| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/fibonacci-47f3e1d8194d39d476ea2a80020ae3797395832e.md) | 473 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/keccak-47f3e1d8194d39d476ea2a80020ae3797395832e.md) | 7,300 |  14,365,133 |  1,520 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/sha2_bench-47f3e1d8194d39d476ea2a80020ae3797395832e.md) | 4,177 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/regex-47f3e1d8194d39d476ea2a80020ae3797395832e.md) | 660 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/ecrecover-47f3e1d8194d39d476ea2a80020ae3797395832e.md) | 223 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/pairing-47f3e1d8194d39d476ea2a80020ae3797395832e.md) | 233 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3107/kitchen_sink-47f3e1d8194d39d476ea2a80020ae3797395832e.md) | 2,033 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/47f3e1d8194d39d476ea2a80020ae3797395832e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31025173484)
