| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/fibonacci-fe0c08ac3695ef70792db244aa3ac170ea2aa6e3.md) | 475 |  4,000,051 |  236 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/keccak-fe0c08ac3695ef70792db244aa3ac170ea2aa6e3.md) | 7,475 |  14,365,133 |  1,601 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/sha2_bench-fe0c08ac3695ef70792db244aa3ac170ea2aa6e3.md) | 4,252 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/regex-fe0c08ac3695ef70792db244aa3ac170ea2aa6e3.md) | 737 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/ecrecover-fe0c08ac3695ef70792db244aa3ac170ea2aa6e3.md) | 210 |  112,210 |  189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/pairing-fe0c08ac3695ef70792db244aa3ac170ea2aa6e3.md) | 253 |  592,827 |  173 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/kitchen_sink-fe0c08ac3695ef70792db244aa3ac170ea2aa6e3.md) | 2,231 |  1,979,971 |  473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fe0c08ac3695ef70792db244aa3ac170ea2aa6e3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34603725715)
