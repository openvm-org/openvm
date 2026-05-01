| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/fibonacci-2d83e8f506928518f25c16e2566ffb78fa15c40e.md) | 3,814 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/keccak-2d83e8f506928518f25c16e2566ffb78fa15c40e.md) | 18,448 |  18,655,329 |  3,290 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/sha2_bench-2d83e8f506928518f25c16e2566ffb78fa15c40e.md) | 9,049 |  14,793,960 |  1,398 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/regex-2d83e8f506928518f25c16e2566ffb78fa15c40e.md) | 1,408 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/ecrecover-2d83e8f506928518f25c16e2566ffb78fa15c40e.md) | 634 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/pairing-2d83e8f506928518f25c16e2566ffb78fa15c40e.md) | 888 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/kitchen_sink-2d83e8f506928518f25c16e2566ffb78fa15c40e.md) | 2,088 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2d83e8f506928518f25c16e2566ffb78fa15c40e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25200669382)
