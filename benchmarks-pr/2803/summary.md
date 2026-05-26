| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/fibonacci-3c8fea1ef0ef3b9e3be71aad81288f45e5008660.md) | 3,776 |  12,000,265 |  918 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/keccak-3c8fea1ef0ef3b9e3be71aad81288f45e5008660.md) | 18,415 |  18,655,329 |  3,255 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/sha2_bench-3c8fea1ef0ef3b9e3be71aad81288f45e5008660.md) | 10,486 |  14,793,960 |  1,491 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/regex-3c8fea1ef0ef3b9e3be71aad81288f45e5008660.md) | 1,408 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/ecrecover-3c8fea1ef0ef3b9e3be71aad81288f45e5008660.md) | 603 |  123,583 |  251 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/pairing-3c8fea1ef0ef3b9e3be71aad81288f45e5008660.md) | 900 |  1,745,757 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/kitchen_sink-3c8fea1ef0ef3b9e3be71aad81288f45e5008660.md) | 1,915 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3c8fea1ef0ef3b9e3be71aad81288f45e5008660

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26473624738)
