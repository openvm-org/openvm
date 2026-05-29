| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2827/fibonacci-044c5375684321d0d80ca72ed8a50e56a7c7bd45.md) | 3,753 |  12,000,265 |  916 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2827/keccak-044c5375684321d0d80ca72ed8a50e56a7c7bd45.md) | 18,583 |  18,655,329 |  3,274 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2827/sha2_bench-044c5375684321d0d80ca72ed8a50e56a7c7bd45.md) | 10,129 |  14,793,960 |  1,440 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2827/regex-044c5375684321d0d80ca72ed8a50e56a7c7bd45.md) | 1,395 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2827/ecrecover-044c5375684321d0d80ca72ed8a50e56a7c7bd45.md) | 615 |  123,583 |  260 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2827/pairing-044c5375684321d0d80ca72ed8a50e56a7c7bd45.md) | 882 |  1,745,757 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2827/kitchen_sink-044c5375684321d0d80ca72ed8a50e56a7c7bd45.md) | 1,896 |  2,579,903 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/044c5375684321d0d80ca72ed8a50e56a7c7bd45

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26611186485)
