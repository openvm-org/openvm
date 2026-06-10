| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-4d410f21fd90a05e7f31627bdd51fbe57a4580a5.md) | 1,640 |  4,000,051 |  520 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-4d410f21fd90a05e7f31627bdd51fbe57a4580a5.md) | 16,494 |  14,365,133 |  3,092 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-4d410f21fd90a05e7f31627bdd51fbe57a4580a5.md) | 10,579 |  11,167,961 |  1,959 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-4d410f21fd90a05e7f31627bdd51fbe57a4580a5.md) | 1,519 |  4,090,656 |  423 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-4d410f21fd90a05e7f31627bdd51fbe57a4580a5.md) | 479 |  112,210 |  307 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-4d410f21fd90a05e7f31627bdd51fbe57a4580a5.md) | 625 |  592,827 |  291 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-4d410f21fd90a05e7f31627bdd51fbe57a4580a5.md) | 3,951 |  1,979,971 |  859 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci_e2e-4d410f21fd90a05e7f31627bdd51fbe57a4580a5.md) | 847 |  4,000,051 |  233 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex_e2e-4d410f21fd90a05e7f31627bdd51fbe57a4580a5.md) | 835 |  4,090,656 |  199 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover_e2e-4d410f21fd90a05e7f31627bdd51fbe57a4580a5.md) | 333 |  112,210 |  151 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing_e2e-4d410f21fd90a05e7f31627bdd51fbe57a4580a5.md) | 404 |  592,827 |  143 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink_e2e-4d410f21fd90a05e7f31627bdd51fbe57a4580a5.md) | 1,942 |  1,979,971 |  371 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4d410f21fd90a05e7f31627bdd51fbe57a4580a5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27297823797)
