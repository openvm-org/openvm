| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/fibonacci-a190ab6f6c05420d1c81b2c3477e11ad3e3b7371.md) | 459 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/keccak-a190ab6f6c05420d1c81b2c3477e11ad3e3b7371.md) | 7,277 |  14,365,133 |  1,547 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/sha2_bench-a190ab6f6c05420d1c81b2c3477e11ad3e3b7371.md) | 4,743 |  11,167,961 |  538 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/regex-a190ab6f6c05420d1c81b2c3477e11ad3e3b7371.md) | 657 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/ecrecover-a190ab6f6c05420d1c81b2c3477e11ad3e3b7371.md) | 229 |  112,210 |  189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/pairing-a190ab6f6c05420d1c81b2c3477e11ad3e3b7371.md) | 296 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3069/kitchen_sink-a190ab6f6c05420d1c81b2c3477e11ad3e3b7371.md) | 2,671 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a190ab6f6c05420d1c81b2c3477e11ad3e3b7371

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30132933205)
