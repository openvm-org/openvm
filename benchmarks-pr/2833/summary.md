| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/fibonacci-fd7f7d860558ac5539a9dd5d283d9d35fa864a97.md) | 3,717 |  12,000,265 |  902 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/keccak-fd7f7d860558ac5539a9dd5d283d9d35fa864a97.md) | 18,553 |  18,655,329 |  3,266 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/sha2_bench-fd7f7d860558ac5539a9dd5d283d9d35fa864a97.md) | 10,165 |  14,793,960 |  1,464 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/regex-fd7f7d860558ac5539a9dd5d283d9d35fa864a97.md) | 1,406 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/ecrecover-fd7f7d860558ac5539a9dd5d283d9d35fa864a97.md) | 601 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/pairing-fd7f7d860558ac5539a9dd5d283d9d35fa864a97.md) | 888 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/kitchen_sink-fd7f7d860558ac5539a9dd5d283d9d35fa864a97.md) | 1,912 |  2,579,903 |  417 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fd7f7d860558ac5539a9dd5d283d9d35fa864a97

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26974751862)
