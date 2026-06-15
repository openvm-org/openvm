| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/fibonacci-4b5977f8da2d4403a8a4ffb480016457fc734569.md) | 3,044 |  12,000,265 |  671 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/keccak-4b5977f8da2d4403a8a4ffb480016457fc734569.md) | 16,277 |  18,655,329 |  3,023 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/sha2_bench-4b5977f8da2d4403a8a4ffb480016457fc734569.md) | 9,221 |  14,793,960 |  1,127 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/regex-4b5977f8da2d4403a8a4ffb480016457fc734569.md) | 1,168 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/ecrecover-4b5977f8da2d4403a8a4ffb480016457fc734569.md) | 602 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/pairing-4b5977f8da2d4403a8a4ffb480016457fc734569.md) | 919 |  1,745,757 |  300 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/kitchen_sink-4b5977f8da2d4403a8a4ffb480016457fc734569.md) | 4,086 |  2,579,903 |  874 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4b5977f8da2d4403a8a4ffb480016457fc734569

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27580285968)
