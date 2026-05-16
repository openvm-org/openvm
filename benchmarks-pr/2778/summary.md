| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-0c7dbcd17c57bc6543a51a290609e8d974c23f6b.md) | 1,443 |  4,000,051 |  445 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-0c7dbcd17c57bc6543a51a290609e8d974c23f6b.md) | 13,278 |  14,365,133 |  2,209 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-0c7dbcd17c57bc6543a51a290609e8d974c23f6b.md) | 9,006 |  11,167,961 |  1,414 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-0c7dbcd17c57bc6543a51a290609e8d974c23f6b.md) | 1,340 |  4,090,656 |  350 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-0c7dbcd17c57bc6543a51a290609e8d974c23f6b.md) | 471 |  112,210 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-0c7dbcd17c57bc6543a51a290609e8d974c23f6b.md) | 597 |  592,827 |  257 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-0c7dbcd17c57bc6543a51a290609e8d974c23f6b.md) | 2,224 |  1,979,971 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0c7dbcd17c57bc6543a51a290609e8d974c23f6b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25969833654)
