| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/fibonacci-338bf2a5ced6d05677f22ad6dd398d6d323f968d.md) | 845 |  4,000,051 |  386 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/keccak-338bf2a5ced6d05677f22ad6dd398d6d323f968d.md) | 15,337 |  14,365,133 |  3,032 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/sha2_bench-338bf2a5ced6d05677f22ad6dd398d6d323f968d.md) | 7,993 |  11,167,961 |  998 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/regex-338bf2a5ced6d05677f22ad6dd398d6d323f968d.md) | 1,033 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/ecrecover-338bf2a5ced6d05677f22ad6dd398d6d323f968d.md) | 306 |  112,210 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/pairing-338bf2a5ced6d05677f22ad6dd398d6d323f968d.md) | 449 |  592,827 |  297 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/kitchen_sink-338bf2a5ced6d05677f22ad6dd398d6d323f968d.md) | 3,697 |  1,979,971 |  859 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/338bf2a5ced6d05677f22ad6dd398d6d323f968d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28979647243)
