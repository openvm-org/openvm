| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/fibonacci-0e17244b38512a6fda73a4b11c32a82bc525f759.md) | 3,730 |  12,000,265 |  903 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/keccak-0e17244b38512a6fda73a4b11c32a82bc525f759.md) | 18,510 |  18,655,329 |  3,261 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/sha2_bench-0e17244b38512a6fda73a4b11c32a82bc525f759.md) | 10,229 |  14,793,960 |  1,463 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/regex-0e17244b38512a6fda73a4b11c32a82bc525f759.md) | 1,386 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/ecrecover-0e17244b38512a6fda73a4b11c32a82bc525f759.md) | 603 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/pairing-0e17244b38512a6fda73a4b11c32a82bc525f759.md) | 889 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/kitchen_sink-0e17244b38512a6fda73a4b11c32a82bc525f759.md) | 1,906 |  2,579,903 |  408 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0e17244b38512a6fda73a4b11c32a82bc525f759

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26249956841)
