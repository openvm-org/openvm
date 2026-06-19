| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/fibonacci-0bdc24f2f65c95a6be7d555b1c7929dca04e892b.md) | 3,049 |  12,000,265 |  670 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/keccak-0bdc24f2f65c95a6be7d555b1c7929dca04e892b.md) | 16,545 |  18,655,329 |  3,065 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/sha2_bench-0bdc24f2f65c95a6be7d555b1c7929dca04e892b.md) | 9,226 |  14,793,960 |  1,127 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/regex-0bdc24f2f65c95a6be7d555b1c7929dca04e892b.md) | 1,157 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/ecrecover-0bdc24f2f65c95a6be7d555b1c7929dca04e892b.md) | 611 |  123,583 |  292 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/pairing-0bdc24f2f65c95a6be7d555b1c7929dca04e892b.md) | 937 |  1,745,757 |  306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/kitchen_sink-0bdc24f2f65c95a6be7d555b1c7929dca04e892b.md) | 4,088 |  2,579,903 |  873 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0bdc24f2f65c95a6be7d555b1c7929dca04e892b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27853297933)
