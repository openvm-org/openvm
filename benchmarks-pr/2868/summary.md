| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2868/fibonacci-b7ff1268f83f678e580cb9ee9e28209bf3e49cc8.md) | 4,037 |  12,000,265 |  1,163 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2868/keccak-b7ff1268f83f678e580cb9ee9e28209bf3e49cc8.md) | 21,766 |  18,655,329 |  4,617 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2868/sha2_bench-b7ff1268f83f678e580cb9ee9e28209bf3e49cc8.md) | 9,656 |  14,793,960 |  1,842 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2868/regex-b7ff1268f83f678e580cb9ee9e28209bf3e49cc8.md) | 1,508 |  4,137,067 |  430 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2868/ecrecover-b7ff1268f83f678e580cb9ee9e28209bf3e49cc8.md) | 611 |  123,583 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2868/pairing-b7ff1268f83f678e580cb9ee9e28209bf3e49cc8.md) | 939 |  1,745,757 |  311 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2868/kitchen_sink-b7ff1268f83f678e580cb9ee9e28209bf3e49cc8.md) | 4,143 |  2,579,903 |  885 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b7ff1268f83f678e580cb9ee9e28209bf3e49cc8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27291358238)
