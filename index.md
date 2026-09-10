| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-113eca64090f1224842f9aa4173ece49e546fb8c.md) | 1,675 |  12,000,265 |  372 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-113eca64090f1224842f9aa4173ece49e546fb8c.md) | 9,478 |  18,655,329 |  1,540 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-113eca64090f1224842f9aa4173ece49e546fb8c.md) | 5,243 |  14,793,960 |  588 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-113eca64090f1224842f9aa4173ece49e546fb8c.md) | 692 |  4,137,067 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-113eca64090f1224842f9aa4173ece49e546fb8c.md) | 428 |  123,583 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-113eca64090f1224842f9aa4173ece49e546fb8c.md) | 589 |  1,745,757 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-113eca64090f1224842f9aa4173ece49e546fb8c.md) | 2,288 |  2,579,903 |  497 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/113eca64090f1224842f9aa4173ece49e546fb8c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33780823929)
