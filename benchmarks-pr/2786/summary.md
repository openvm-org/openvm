| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2786/fibonacci-ab0d743b8afd187728fc710468230c80c1f33b10.md) | 3,759 |  12,000,265 |  912 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2786/keccak-ab0d743b8afd187728fc710468230c80c1f33b10.md) | 18,389 |  18,655,329 |  3,243 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2786/sha2_bench-ab0d743b8afd187728fc710468230c80c1f33b10.md) | 10,119 |  14,793,960 |  1,449 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2786/regex-ab0d743b8afd187728fc710468230c80c1f33b10.md) | 1,388 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2786/ecrecover-ab0d743b8afd187728fc710468230c80c1f33b10.md) | 596 |  123,583 |  251 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2786/pairing-ab0d743b8afd187728fc710468230c80c1f33b10.md) | 885 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2786/kitchen_sink-ab0d743b8afd187728fc710468230c80c1f33b10.md) | 1,891 |  2,579,903 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ab0d743b8afd187728fc710468230c80c1f33b10

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25940073814)
