| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/fibonacci-41f45e1295d0448d82183247adc5823f5d3cae6e.md) | 412 |  4,000,051 |  236 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/keccak-41f45e1295d0448d82183247adc5823f5d3cae6e.md) | 8,740 |  14,365,133 |  1,546 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/sha2_bench-41f45e1295d0448d82183247adc5823f5d3cae6e.md) | 4,258 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/regex-41f45e1295d0448d82183247adc5823f5d3cae6e.md) | 582 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/ecrecover-41f45e1295d0448d82183247adc5823f5d3cae6e.md) | 218 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/pairing-41f45e1295d0448d82183247adc5823f5d3cae6e.md) | 284 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/kitchen_sink-41f45e1295d0448d82183247adc5823f5d3cae6e.md) | 1,911 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/41f45e1295d0448d82183247adc5823f5d3cae6e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29658421309)
