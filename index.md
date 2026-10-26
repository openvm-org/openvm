| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-e19e0c95ce08470abac3193dd7284621ec29206e.md) | 1,679 |  12,000,265 |  371 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-e19e0c95ce08470abac3193dd7284621ec29206e.md) | 9,512 |  18,655,329 |  1,539 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-e19e0c95ce08470abac3193dd7284621ec29206e.md) | 5,216 |  14,793,960 |  584 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-e19e0c95ce08470abac3193dd7284621ec29206e.md) | 682 |  4,137,067 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-e19e0c95ce08470abac3193dd7284621ec29206e.md) | 427 |  123,583 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-e19e0c95ce08470abac3193dd7284621ec29206e.md) | 582 |  1,745,757 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-e19e0c95ce08470abac3193dd7284621ec29206e.md) | 2,287 |  2,579,903 |  493 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e19e0c95ce08470abac3193dd7284621ec29206e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/40367704769)
