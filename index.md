| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-922470a804cbb40ce4fa81647db64edd4643a410.md) | 1,707 |  12,000,265 |  375 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-922470a804cbb40ce4fa81647db64edd4643a410.md) | 9,536 |  18,655,329 |  1,549 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-922470a804cbb40ce4fa81647db64edd4643a410.md) | 5,241 |  14,793,960 |  586 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-922470a804cbb40ce4fa81647db64edd4643a410.md) | 699 |  4,137,067 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-922470a804cbb40ce4fa81647db64edd4643a410.md) | 427 |  123,583 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-922470a804cbb40ce4fa81647db64edd4643a410.md) | 565 |  1,745,757 |  193 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-922470a804cbb40ce4fa81647db64edd4643a410.md) | 2,307 |  2,579,903 |  498 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/922470a804cbb40ce4fa81647db64edd4643a410

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34995054692)
