| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-15a7ab6baed03d75050dbef2bbad4b4e98fb8dba.md) | 3,034 |  12,000,265 |  671 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-15a7ab6baed03d75050dbef2bbad4b4e98fb8dba.md) | 16,329 |  18,655,329 |  3,028 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-15a7ab6baed03d75050dbef2bbad4b4e98fb8dba.md) | 9,137 |  14,793,960 |  1,123 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-15a7ab6baed03d75050dbef2bbad4b4e98fb8dba.md) | 1,167 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-15a7ab6baed03d75050dbef2bbad4b4e98fb8dba.md) | 598 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-15a7ab6baed03d75050dbef2bbad4b4e98fb8dba.md) | 931 |  1,745,757 |  308 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-15a7ab6baed03d75050dbef2bbad4b4e98fb8dba.md) | 4,125 |  2,579,903 |  880 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/15a7ab6baed03d75050dbef2bbad4b4e98fb8dba

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29072061227)
