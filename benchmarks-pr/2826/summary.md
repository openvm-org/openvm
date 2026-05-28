| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/fibonacci-00b36887a1c298ce05c88523328e95e924386430.md) | 3,793 |  12,000,265 |  926 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/keccak-00b36887a1c298ce05c88523328e95e924386430.md) | 18,445 |  18,655,329 |  3,254 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/sha2_bench-00b36887a1c298ce05c88523328e95e924386430.md) | 10,249 |  14,793,960 |  1,461 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/regex-00b36887a1c298ce05c88523328e95e924386430.md) | 1,410 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/ecrecover-00b36887a1c298ce05c88523328e95e924386430.md) | 596 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/pairing-00b36887a1c298ce05c88523328e95e924386430.md) | 881 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/kitchen_sink-00b36887a1c298ce05c88523328e95e924386430.md) | 1,896 |  2,579,903 |  411 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/00b36887a1c298ce05c88523328e95e924386430

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26600528588)
