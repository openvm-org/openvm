| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-314831fec6cded624818e06d4462fa80d9e310da.md) | 406 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-314831fec6cded624818e06d4462fa80d9e310da.md) | 8,742 |  14,365,133 |  1,538 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-314831fec6cded624818e06d4462fa80d9e310da.md) | 4,179 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-314831fec6cded624818e06d4462fa80d9e310da.md) | 570 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-314831fec6cded624818e06d4462fa80d9e310da.md) | 227 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-314831fec6cded624818e06d4462fa80d9e310da.md) | 294 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-314831fec6cded624818e06d4462fa80d9e310da.md) | 1,916 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/314831fec6cded624818e06d4462fa80d9e310da

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29560097045)
