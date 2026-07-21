| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/fibonacci-cafed8aca460f6108f979665b25a10c775d2df66.md) | 418 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/keccak-cafed8aca460f6108f979665b25a10c775d2df66.md) | 8,681 |  14,365,133 |  1,551 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/sha2_bench-cafed8aca460f6108f979665b25a10c775d2df66.md) | 4,155 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/regex-cafed8aca460f6108f979665b25a10c775d2df66.md) | 556 |  4,090,656 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/ecrecover-cafed8aca460f6108f979665b25a10c775d2df66.md) | 219 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/pairing-cafed8aca460f6108f979665b25a10c775d2df66.md) | 283 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/kitchen_sink-cafed8aca460f6108f979665b25a10c775d2df66.md) | 1,910 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cafed8aca460f6108f979665b25a10c775d2df66

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29818327229)
