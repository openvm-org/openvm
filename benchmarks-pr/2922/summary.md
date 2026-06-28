| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/fibonacci-9c4933c46c8adbb6eb5906591db6c8cbd50eb5e5.md) | 1,019 |  4,000,051 |  390 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/keccak-9c4933c46c8adbb6eb5906591db6c8cbd50eb5e5.md) | 15,508 |  14,365,133 |  3,032 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/sha2_bench-9c4933c46c8adbb6eb5906591db6c8cbd50eb5e5.md) | 7,798 |  11,167,961 |  993 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/regex-9c4933c46c8adbb6eb5906591db6c8cbd50eb5e5.md) | 1,021 |  4,090,656 |  302 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/ecrecover-9c4933c46c8adbb6eb5906591db6c8cbd50eb5e5.md) | 443 |  112,210 |  296 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/pairing-9c4933c46c8adbb6eb5906591db6c8cbd50eb5e5.md) | 548 |  592,827 |  297 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/kitchen_sink-9c4933c46c8adbb6eb5906591db6c8cbd50eb5e5.md) | 3,769 |  1,979,971 |  862 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9c4933c46c8adbb6eb5906591db6c8cbd50eb5e5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28324842506)
