| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/fibonacci-331c449318d85c153796245d85ba10656dca150e.md) | 1,657 |  4,000,051 |  527 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/keccak-331c449318d85c153796245d85ba10656dca150e.md) | 16,287 |  14,365,133 |  3,030 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/sha2_bench-331c449318d85c153796245d85ba10656dca150e.md) | 10,492 |  11,167,961 |  1,950 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/regex-331c449318d85c153796245d85ba10656dca150e.md) | 1,525 |  4,090,656 |  427 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/ecrecover-331c449318d85c153796245d85ba10656dca150e.md) | 484 |  112,210 |  310 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/pairing-331c449318d85c153796245d85ba10656dca150e.md) | 618 |  592,827 |  292 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2890/kitchen_sink-331c449318d85c153796245d85ba10656dca150e.md) | 4,022 |  1,979,971 |  879 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/331c449318d85c153796245d85ba10656dca150e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27709413793)
