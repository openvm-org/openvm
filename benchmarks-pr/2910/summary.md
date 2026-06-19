| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/fibonacci-3531528dc7b4db44e5281a55ab2832bbd48029fa.md) | 3,048 |  12,000,265 |  672 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/keccak-3531528dc7b4db44e5281a55ab2832bbd48029fa.md) | 16,287 |  18,655,329 |  3,022 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/sha2_bench-3531528dc7b4db44e5281a55ab2832bbd48029fa.md) | 9,127 |  14,793,960 |  1,111 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/regex-3531528dc7b4db44e5281a55ab2832bbd48029fa.md) | 1,167 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/ecrecover-3531528dc7b4db44e5281a55ab2832bbd48029fa.md) | 602 |  123,583 |  286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/pairing-3531528dc7b4db44e5281a55ab2832bbd48029fa.md) | 954 |  1,745,757 |  318 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/kitchen_sink-3531528dc7b4db44e5281a55ab2832bbd48029fa.md) | 4,117 |  2,579,903 |  883 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3531528dc7b4db44e5281a55ab2832bbd48029fa

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27843576996)
