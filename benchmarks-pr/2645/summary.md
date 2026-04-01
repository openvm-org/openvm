| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2645/fibonacci-eeec1125f88b6bb13e6e6ba04ec9badc3e47a245.md) | 3,833 |  12,000,265 |  945 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2645/keccak-eeec1125f88b6bb13e6e6ba04ec9badc3e47a245.md) | 18,574 |  18,655,329 |  3,322 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2645/regex-eeec1125f88b6bb13e6e6ba04ec9badc3e47a245.md) | 1,441 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2645/ecrecover-eeec1125f88b6bb13e6e6ba04ec9badc3e47a245.md) | 651 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2645/pairing-eeec1125f88b6bb13e6e6ba04ec9badc3e47a245.md) | 908 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2645/kitchen_sink-eeec1125f88b6bb13e6e6ba04ec9badc3e47a245.md) | 2,277 |  2,579,903 |  440 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/eeec1125f88b6bb13e6e6ba04ec9badc3e47a245

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23875273586)
