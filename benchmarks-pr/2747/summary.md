| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2747/fibonacci-dace11f25f8560f011a31a17a50589e88cf57c61.md) | 3,863 |  12,000,265 |  964 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2747/keccak-dace11f25f8560f011a31a17a50589e88cf57c61.md) | 18,665 |  18,655,329 |  3,329 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2747/sha2_bench-dace11f25f8560f011a31a17a50589e88cf57c61.md) | 8,943 |  14,793,960 |  1,390 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2747/regex-dace11f25f8560f011a31a17a50589e88cf57c61.md) | 1,415 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2747/ecrecover-dace11f25f8560f011a31a17a50589e88cf57c61.md) | 641 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2747/pairing-dace11f25f8560f011a31a17a50589e88cf57c61.md) | 902 |  1,745,757 |  278 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2747/kitchen_sink-dace11f25f8560f011a31a17a50589e88cf57c61.md) | 2,089 |  2,579,903 |  431 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dace11f25f8560f011a31a17a50589e88cf57c61

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24856205415)
