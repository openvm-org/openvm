| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/fibonacci-99ab14710a9ee87f28dc27e118c75ce98b9282d2.md) | 4,001 |  12,000,265 |  1,155 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/keccak-99ab14710a9ee87f28dc27e118c75ce98b9282d2.md) | 21,886 |  18,655,329 |  4,637 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/sha2_bench-99ab14710a9ee87f28dc27e118c75ce98b9282d2.md) | 9,467 |  14,793,960 |  1,818 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/regex-99ab14710a9ee87f28dc27e118c75ce98b9282d2.md) | 1,497 |  4,137,067 |  428 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/ecrecover-99ab14710a9ee87f28dc27e118c75ce98b9282d2.md) | 601 |  123,583 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/pairing-99ab14710a9ee87f28dc27e118c75ce98b9282d2.md) | 929 |  1,745,757 |  307 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/kitchen_sink-99ab14710a9ee87f28dc27e118c75ce98b9282d2.md) | 4,114 |  2,579,903 |  874 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/99ab14710a9ee87f28dc27e118c75ce98b9282d2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27188345740)
