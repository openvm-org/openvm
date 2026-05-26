| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/fibonacci-e149bf69a1c6835a528ef7491803022147e39b4c.md) | 3,737 |  12,000,265 |  912 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/keccak-e149bf69a1c6835a528ef7491803022147e39b4c.md) | 18,565 |  18,655,329 |  3,268 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/sha2_bench-e149bf69a1c6835a528ef7491803022147e39b4c.md) | 10,184 |  14,793,960 |  1,461 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/regex-e149bf69a1c6835a528ef7491803022147e39b4c.md) | 1,401 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/ecrecover-e149bf69a1c6835a528ef7491803022147e39b4c.md) | 605 |  123,583 |  255 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/pairing-e149bf69a1c6835a528ef7491803022147e39b4c.md) | 891 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2815/kitchen_sink-e149bf69a1c6835a528ef7491803022147e39b4c.md) | 1,897 |  2,579,903 |  411 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e149bf69a1c6835a528ef7491803022147e39b4c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26468054199)
