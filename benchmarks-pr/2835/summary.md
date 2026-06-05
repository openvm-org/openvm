| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/fibonacci-f4244423737823cd4c1aec6a8372876e4a846a37.md) | 3,818 |  12,000,265 |  933 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/keccak-f4244423737823cd4c1aec6a8372876e4a846a37.md) | 18,277 |  18,655,329 |  3,318 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/sha2_bench-f4244423737823cd4c1aec6a8372876e4a846a37.md) | 9,969 |  14,793,960 |  1,460 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/regex-f4244423737823cd4c1aec6a8372876e4a846a37.md) | 1,390 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/ecrecover-f4244423737823cd4c1aec6a8372876e4a846a37.md) | 600 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/pairing-f4244423737823cd4c1aec6a8372876e4a846a37.md) | 881 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/kitchen_sink-f4244423737823cd4c1aec6a8372876e4a846a37.md) | 1,867 |  2,579,903 |  417 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f4244423737823cd4c1aec6a8372876e4a846a37

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27022133071)
