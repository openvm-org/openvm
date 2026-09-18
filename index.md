| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-f08bf2836409f3c0a5f6b6cfe73eb177a8a3e8c8.md) | 1,679 |  12,000,265 |  371 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-f08bf2836409f3c0a5f6b6cfe73eb177a8a3e8c8.md) | 9,520 |  18,655,329 |  1,539 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-f08bf2836409f3c0a5f6b6cfe73eb177a8a3e8c8.md) | 5,320 |  14,793,960 |  594 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-f08bf2836409f3c0a5f6b6cfe73eb177a8a3e8c8.md) | 698 |  4,137,067 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-f08bf2836409f3c0a5f6b6cfe73eb177a8a3e8c8.md) | 439 |  123,583 |  190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-f08bf2836409f3c0a5f6b6cfe73eb177a8a3e8c8.md) | 577 |  1,745,757 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-f08bf2836409f3c0a5f6b6cfe73eb177a8a3e8c8.md) | 2,322 |  2,579,903 |  496 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f08bf2836409f3c0a5f6b6cfe73eb177a8a3e8c8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/35359825802)
