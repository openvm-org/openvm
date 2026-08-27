| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-98f0cc83ec5330f08b5943df87c8430abe8de705.md) | 1,647 |  12,000,265 |  367 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-98f0cc83ec5330f08b5943df87c8430abe8de705.md) | 9,652 |  18,655,329 |  1,566 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-98f0cc83ec5330f08b5943df87c8430abe8de705.md) | 5,373 |  14,793,960 |  596 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-98f0cc83ec5330f08b5943df87c8430abe8de705.md) | 700 |  4,137,067 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-98f0cc83ec5330f08b5943df87c8430abe8de705.md) | 425 |  123,583 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-98f0cc83ec5330f08b5943df87c8430abe8de705.md) | 579 |  1,745,757 |  198 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-98f0cc83ec5330f08b5943df87c8430abe8de705.md) | 2,293 |  2,579,903 |  493 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/98f0cc83ec5330f08b5943df87c8430abe8de705

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33100899857)
