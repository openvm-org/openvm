| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-ee9a87629b7848fafe73f016f88e900a519baac8.md) | 869 |  4,000,051 |  395 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-ee9a87629b7848fafe73f016f88e900a519baac8.md) | 15,594 |  14,365,133 |  3,024 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-ee9a87629b7848fafe73f016f88e900a519baac8.md) | 7,845 |  11,167,961 |  1,000 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-ee9a87629b7848fafe73f016f88e900a519baac8.md) | 1,018 |  4,090,656 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-ee9a87629b7848fafe73f016f88e900a519baac8.md) | 308 |  112,210 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-ee9a87629b7848fafe73f016f88e900a519baac8.md) | 441 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-ee9a87629b7848fafe73f016f88e900a519baac8.md) | 3,732 |  1,979,971 |  866 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ee9a87629b7848fafe73f016f88e900a519baac8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29127197351)
