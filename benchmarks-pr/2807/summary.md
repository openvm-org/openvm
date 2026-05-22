| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/fibonacci-73d348425fc0f7ee80f6690fb790ec850590f9a9.md) | 3,695 |  12,000,265 |  907 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/keccak-73d348425fc0f7ee80f6690fb790ec850590f9a9.md) | 18,533 |  18,655,329 |  3,267 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/sha2_bench-73d348425fc0f7ee80f6690fb790ec850590f9a9.md) | 10,260 |  14,793,960 |  1,470 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/regex-73d348425fc0f7ee80f6690fb790ec850590f9a9.md) | 1,415 |  4,137,067 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/ecrecover-73d348425fc0f7ee80f6690fb790ec850590f9a9.md) | 602 |  123,583 |  251 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/pairing-73d348425fc0f7ee80f6690fb790ec850590f9a9.md) | 887 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/kitchen_sink-73d348425fc0f7ee80f6690fb790ec850590f9a9.md) | 1,903 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/73d348425fc0f7ee80f6690fb790ec850590f9a9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26289492808)
