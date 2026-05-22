| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/fibonacci-5a1159cc177adb9722c5b7a8de0a8e1ef8271528.md) | 3,754 |  12,000,265 |  920 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/keccak-5a1159cc177adb9722c5b7a8de0a8e1ef8271528.md) | 18,542 |  18,655,329 |  3,262 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/sha2_bench-5a1159cc177adb9722c5b7a8de0a8e1ef8271528.md) | 10,181 |  14,793,960 |  1,457 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/regex-5a1159cc177adb9722c5b7a8de0a8e1ef8271528.md) | 1,389 |  4,137,067 |  350 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/ecrecover-5a1159cc177adb9722c5b7a8de0a8e1ef8271528.md) | 592 |  123,583 |  251 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/pairing-5a1159cc177adb9722c5b7a8de0a8e1ef8271528.md) | 889 |  1,745,757 |  265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/kitchen_sink-5a1159cc177adb9722c5b7a8de0a8e1ef8271528.md) | 1,907 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5a1159cc177adb9722c5b7a8de0a8e1ef8271528

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26292221088)
