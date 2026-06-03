| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/fibonacci-cbfce17e0f2a2d2703afd983b4a028c5c4d72524.md) | 3,752 |  12,000,265 |  913 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/keccak-cbfce17e0f2a2d2703afd983b4a028c5c4d72524.md) | 18,571 |  18,655,329 |  3,265 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/sha2_bench-cbfce17e0f2a2d2703afd983b4a028c5c4d72524.md) | 10,181 |  14,793,960 |  1,455 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/regex-cbfce17e0f2a2d2703afd983b4a028c5c4d72524.md) | 1,440 |  4,137,067 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/ecrecover-cbfce17e0f2a2d2703afd983b4a028c5c4d72524.md) | 604 |  123,583 |  255 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/pairing-cbfce17e0f2a2d2703afd983b4a028c5c4d72524.md) | 896 |  1,745,757 |  267 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2834/kitchen_sink-cbfce17e0f2a2d2703afd983b4a028c5c4d72524.md) | 1,914 |  2,579,903 |  416 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cbfce17e0f2a2d2703afd983b4a028c5c4d72524

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26894822367)
