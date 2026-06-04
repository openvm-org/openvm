| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-7e4e1c1ff82339d6e5fcea5bed3413757f969016.md) | 1,542 |  4,000,051 |  431 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-7e4e1c1ff82339d6e5fcea5bed3413757f969016.md) | 13,968 |  14,365,133 |  2,402 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-7e4e1c1ff82339d6e5fcea5bed3413757f969016.md) | 8,914 |  11,167,961 |  1,410 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-7e4e1c1ff82339d6e5fcea5bed3413757f969016.md) | 1,466 |  4,090,656 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-7e4e1c1ff82339d6e5fcea5bed3413757f969016.md) | 469 |  112,210 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-7e4e1c1ff82339d6e5fcea5bed3413757f969016.md) | 583 |  592,827 |  243 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-7e4e1c1ff82339d6e5fcea5bed3413757f969016.md) | 3,829 |  1,979,971 |  957 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci_e2e-7e4e1c1ff82339d6e5fcea5bed3413757f969016.md) | 807 |  4,000,051 |  200 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex_e2e-7e4e1c1ff82339d6e5fcea5bed3413757f969016.md) | 868 |  4,090,656 |  170 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover_e2e-7e4e1c1ff82339d6e5fcea5bed3413757f969016.md) | 325 |  112,210 |  134 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing_e2e-7e4e1c1ff82339d6e5fcea5bed3413757f969016.md) | 391 |  592,827 |  128 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink_e2e-7e4e1c1ff82339d6e5fcea5bed3413757f969016.md) | 2,038 |  1,979,971 |  392 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7e4e1c1ff82339d6e5fcea5bed3413757f969016

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26973658914)
