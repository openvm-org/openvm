| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3060/fibonacci-ab7a54b3db58d58ba6a32a5e2d1940c7dca67db7.md) | 469 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3060/keccak-ab7a54b3db58d58ba6a32a5e2d1940c7dca67db7.md) | 7,319 |  14,365,133 |  1,556 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3060/sha2_bench-ab7a54b3db58d58ba6a32a5e2d1940c7dca67db7.md) | 4,727 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3060/regex-ab7a54b3db58d58ba6a32a5e2d1940c7dca67db7.md) | 669 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3060/ecrecover-ab7a54b3db58d58ba6a32a5e2d1940c7dca67db7.md) | 228 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3060/pairing-ab7a54b3db58d58ba6a32a5e2d1940c7dca67db7.md) | 325 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3060/kitchen_sink-ab7a54b3db58d58ba6a32a5e2d1940c7dca67db7.md) | 2,687 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ab7a54b3db58d58ba6a32a5e2d1940c7dca67db7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29939067350)
