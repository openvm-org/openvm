| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/fibonacci-01551e9a87209381824ecaee5ed99505ecb175f0.md) | 3,788 |  12,000,265 |  924 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/keccak-01551e9a87209381824ecaee5ed99505ecb175f0.md) | 18,744 |  18,655,329 |  3,304 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/sha2_bench-01551e9a87209381824ecaee5ed99505ecb175f0.md) | 10,170 |  14,793,960 |  1,450 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/regex-01551e9a87209381824ecaee5ed99505ecb175f0.md) | 1,394 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/ecrecover-01551e9a87209381824ecaee5ed99505ecb175f0.md) | 608 |  123,583 |  254 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/pairing-01551e9a87209381824ecaee5ed99505ecb175f0.md) | 887 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/kitchen_sink-01551e9a87209381824ecaee5ed99505ecb175f0.md) | 1,905 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/01551e9a87209381824ecaee5ed99505ecb175f0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26898436584)
