| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/fibonacci-293c488325d91a561b3bde8e063d07f2e5b20b3e.md) | 3,821 |  12,000,265 |  961 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/keccak-293c488325d91a561b3bde8e063d07f2e5b20b3e.md) | 19,245 |  18,655,329 |  3,386 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/sha2_bench-293c488325d91a561b3bde8e063d07f2e5b20b3e.md) | 9,166 |  14,793,960 |  1,414 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/regex-293c488325d91a561b3bde8e063d07f2e5b20b3e.md) | 1,417 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/ecrecover-293c488325d91a561b3bde8e063d07f2e5b20b3e.md) | 645 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/pairing-293c488325d91a561b3bde8e063d07f2e5b20b3e.md) | 918 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/kitchen_sink-293c488325d91a561b3bde8e063d07f2e5b20b3e.md) | 2,060 |  2,579,903 |  443 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/293c488325d91a561b3bde8e063d07f2e5b20b3e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25814754001)
