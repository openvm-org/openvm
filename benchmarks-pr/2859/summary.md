| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/fibonacci-8e452fc5a1c5261f3ef68d9ef5b3430e7520dc8b.md) | 3,766 |  12,000,265 |  932 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/keccak-8e452fc5a1c5261f3ef68d9ef5b3430e7520dc8b.md) | 18,110 |  18,655,329 |  3,295 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/sha2_bench-8e452fc5a1c5261f3ef68d9ef5b3430e7520dc8b.md) | 9,928 |  14,793,960 |  1,456 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/regex-8e452fc5a1c5261f3ef68d9ef5b3430e7520dc8b.md) | 1,383 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/ecrecover-8e452fc5a1c5261f3ef68d9ef5b3430e7520dc8b.md) | 599 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/pairing-8e452fc5a1c5261f3ef68d9ef5b3430e7520dc8b.md) | 883 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/kitchen_sink-8e452fc5a1c5261f3ef68d9ef5b3430e7520dc8b.md) | 3,823 |  2,579,903 |  941 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8e452fc5a1c5261f3ef68d9ef5b3430e7520dc8b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27167576631)
