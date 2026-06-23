| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2925/fibonacci-0516e739c2a8c927b5dc06e1e53096c98debef6b.md) | 1,038 |  4,000,051 |  390 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2925/keccak-0516e739c2a8c927b5dc06e1e53096c98debef6b.md) | 16,284 |  14,365,133 |  3,011 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2925/sha2_bench-0516e739c2a8c927b5dc06e1e53096c98debef6b.md) | 8,153 |  11,167,961 |  995 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2925/regex-0516e739c2a8c927b5dc06e1e53096c98debef6b.md) | 1,227 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2925/ecrecover-0516e739c2a8c927b5dc06e1e53096c98debef6b.md) | 437 |  112,210 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2925/pairing-0516e739c2a8c927b5dc06e1e53096c98debef6b.md) | 603 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2925/kitchen_sink-0516e739c2a8c927b5dc06e1e53096c98debef6b.md) | 3,852 |  1,979,971 |  853 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0516e739c2a8c927b5dc06e1e53096c98debef6b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28048366427)
