| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/fibonacci-a687fddfa84cbc79f25e20d5bf3091ba66eaa073.md) | 3,093 |  12,000,265 |  682 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/keccak-a687fddfa84cbc79f25e20d5bf3091ba66eaa073.md) | 16,599 |  18,655,329 |  3,087 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/sha2_bench-a687fddfa84cbc79f25e20d5bf3091ba66eaa073.md) | 9,134 |  14,793,960 |  1,114 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/regex-a687fddfa84cbc79f25e20d5bf3091ba66eaa073.md) | 1,152 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/ecrecover-a687fddfa84cbc79f25e20d5bf3091ba66eaa073.md) | 599 |  123,583 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/pairing-a687fddfa84cbc79f25e20d5bf3091ba66eaa073.md) | 941 |  1,745,757 |  306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/kitchen_sink-a687fddfa84cbc79f25e20d5bf3091ba66eaa073.md) | 4,126 |  2,579,903 |  880 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a687fddfa84cbc79f25e20d5bf3091ba66eaa073

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27495713873)
