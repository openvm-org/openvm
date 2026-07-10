| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/fibonacci-73552d98e87bbd9477e6888a1d37bc7cd612ae6f.md) | 976 |  4,000,051 |  401 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/keccak-73552d98e87bbd9477e6888a1d37bc7cd612ae6f.md) | 15,883 |  14,365,133 |  3,055 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/sha2_bench-73552d98e87bbd9477e6888a1d37bc7cd612ae6f.md) | 8,349 |  11,167,961 |  1,016 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/regex-73552d98e87bbd9477e6888a1d37bc7cd612ae6f.md) | 1,198 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/ecrecover-73552d98e87bbd9477e6888a1d37bc7cd612ae6f.md) | 436 |  112,210 |  280 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/pairing-73552d98e87bbd9477e6888a1d37bc7cd612ae6f.md) | 582 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/kitchen_sink-73552d98e87bbd9477e6888a1d37bc7cd612ae6f.md) | 3,832 |  1,979,971 |  865 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/73552d98e87bbd9477e6888a1d37bc7cd612ae6f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29099162729)
