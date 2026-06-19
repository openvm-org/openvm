| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2913/fibonacci-5c137cf67ea88a866e785afd0698ad8300f81428.md) | 1,035 |  4,000,051 |  400 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2913/keccak-5c137cf67ea88a866e785afd0698ad8300f81428.md) | 16,338 |  14,365,133 |  3,025 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2913/sha2_bench-5c137cf67ea88a866e785afd0698ad8300f81428.md) | 8,260 |  11,167,961 |  1,010 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2913/regex-5c137cf67ea88a866e785afd0698ad8300f81428.md) | 1,223 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2913/ecrecover-5c137cf67ea88a866e785afd0698ad8300f81428.md) | 438 |  112,210 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2913/pairing-5c137cf67ea88a866e785afd0698ad8300f81428.md) | 595 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2913/kitchen_sink-5c137cf67ea88a866e785afd0698ad8300f81428.md) | 3,859 |  1,979,971 |  851 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5c137cf67ea88a866e785afd0698ad8300f81428

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27846985841)
