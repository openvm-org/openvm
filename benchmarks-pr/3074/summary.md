| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-970453f169c136193210eb339cb1041fad3885d0.md) | 482 |  4,000,051 |  216 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-970453f169c136193210eb339cb1041fad3885d0.md) | 7,336 |  14,365,133 |  1,488 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-970453f169c136193210eb339cb1041fad3885d0.md) | 4,124 |  11,167,961 |  510 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-970453f169c136193210eb339cb1041fad3885d0.md) | 664 |  4,090,656 |  202 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-970453f169c136193210eb339cb1041fad3885d0.md) | 214 |  112,210 |  171 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-970453f169c136193210eb339cb1041fad3885d0.md) | 229 |  592,827 |  174 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-970453f169c136193210eb339cb1041fad3885d0.md) | 2,011 |  1,979,971 |  446 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/970453f169c136193210eb339cb1041fad3885d0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30386023775)
