| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/fibonacci-e16b82f410363dfe7413aac93e3be2b853837d99.md) | 3,020 |  12,000,265 |  667 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/keccak-e16b82f410363dfe7413aac93e3be2b853837d99.md) | 16,125 |  18,655,329 |  2,998 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/sha2_bench-e16b82f410363dfe7413aac93e3be2b853837d99.md) | 9,149 |  14,793,960 |  1,119 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/regex-e16b82f410363dfe7413aac93e3be2b853837d99.md) | 1,162 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/ecrecover-e16b82f410363dfe7413aac93e3be2b853837d99.md) | 597 |  123,583 |  286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/pairing-e16b82f410363dfe7413aac93e3be2b853837d99.md) | 946 |  1,745,757 |  304 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/kitchen_sink-e16b82f410363dfe7413aac93e3be2b853837d99.md) | 4,096 |  2,579,903 |  882 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e16b82f410363dfe7413aac93e3be2b853837d99

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28076440669)
