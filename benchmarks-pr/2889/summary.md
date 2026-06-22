| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/fibonacci-de6efd91280420e1202c064819e920235bb7d56c.md) | 3,049 |  12,000,265 |  670 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/keccak-de6efd91280420e1202c064819e920235bb7d56c.md) | 16,273 |  18,655,329 |  3,002 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/sha2_bench-de6efd91280420e1202c064819e920235bb7d56c.md) | 9,138 |  14,793,960 |  1,122 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/regex-de6efd91280420e1202c064819e920235bb7d56c.md) | 1,170 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/ecrecover-de6efd91280420e1202c064819e920235bb7d56c.md) | 600 |  123,583 |  291 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/pairing-de6efd91280420e1202c064819e920235bb7d56c.md) | 937 |  1,745,757 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/kitchen_sink-de6efd91280420e1202c064819e920235bb7d56c.md) | 4,106 |  2,579,903 |  877 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/de6efd91280420e1202c064819e920235bb7d56c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27976509218)
