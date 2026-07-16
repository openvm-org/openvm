| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-40d4ae5c495a37bb843ed99fec925fc62fe47bf6.md) | 416 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-40d4ae5c495a37bb843ed99fec925fc62fe47bf6.md) | 8,756 |  14,365,133 |  1,539 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-40d4ae5c495a37bb843ed99fec925fc62fe47bf6.md) | 4,267 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-40d4ae5c495a37bb843ed99fec925fc62fe47bf6.md) | 586 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-40d4ae5c495a37bb843ed99fec925fc62fe47bf6.md) | 221 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-40d4ae5c495a37bb843ed99fec925fc62fe47bf6.md) | 298 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-40d4ae5c495a37bb843ed99fec925fc62fe47bf6.md) | 1,905 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/40d4ae5c495a37bb843ed99fec925fc62fe47bf6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29504423661)
