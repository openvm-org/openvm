| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-0994f54163a29131ac05bfe84f6584b451b2f7fd.md) | 1,677 |  12,000,265 |  367 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-0994f54163a29131ac05bfe84f6584b451b2f7fd.md) | 9,693 |  18,655,329 |  1,547 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-0994f54163a29131ac05bfe84f6584b451b2f7fd.md) | 5,333 |  14,793,960 |  596 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-0994f54163a29131ac05bfe84f6584b451b2f7fd.md) | 704 |  4,137,067 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-0994f54163a29131ac05bfe84f6584b451b2f7fd.md) | 430 |  123,583 |  189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-0994f54163a29131ac05bfe84f6584b451b2f7fd.md) | 563 |  1,745,757 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-0994f54163a29131ac05bfe84f6584b451b2f7fd.md) | 2,306 |  2,579,903 |  493 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0994f54163a29131ac05bfe84f6584b451b2f7fd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/38978209993)
