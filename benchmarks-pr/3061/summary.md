| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/fibonacci-1a669d261f90999c89c3aea89cf2a1272dc43daf.md) | 464 |  4,000,051 |  244 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/keccak-1a669d261f90999c89c3aea89cf2a1272dc43daf.md) | 7,303 |  14,365,133 |  1,545 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/sha2_bench-1a669d261f90999c89c3aea89cf2a1272dc43daf.md) | 4,638 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/regex-1a669d261f90999c89c3aea89cf2a1272dc43daf.md) | 669 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/ecrecover-1a669d261f90999c89c3aea89cf2a1272dc43daf.md) | 229 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/pairing-1a669d261f90999c89c3aea89cf2a1272dc43daf.md) | 319 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/kitchen_sink-1a669d261f90999c89c3aea89cf2a1272dc43daf.md) | 2,685 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1a669d261f90999c89c3aea89cf2a1272dc43daf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30024587529)
