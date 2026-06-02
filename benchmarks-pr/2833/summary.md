| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/fibonacci-615d96cf0bde798057206984975f80ab4bec4152.md) | 3,725 |  12,000,265 |  911 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/keccak-615d96cf0bde798057206984975f80ab4bec4152.md) | 18,513 |  18,655,329 |  3,264 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/sha2_bench-615d96cf0bde798057206984975f80ab4bec4152.md) | 10,129 |  14,793,960 |  1,449 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/regex-615d96cf0bde798057206984975f80ab4bec4152.md) | 1,399 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/ecrecover-615d96cf0bde798057206984975f80ab4bec4152.md) | 599 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/pairing-615d96cf0bde798057206984975f80ab4bec4152.md) | 886 |  1,745,757 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/kitchen_sink-615d96cf0bde798057206984975f80ab4bec4152.md) | 1,915 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/615d96cf0bde798057206984975f80ab4bec4152

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26848483910)
