| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/fibonacci-ae5b4911aa6bad3725938e02b6a4b0612156091d.md) | 3,700 |  12,000,265 |  918 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/keccak-ae5b4911aa6bad3725938e02b6a4b0612156091d.md) | 18,382 |  18,655,329 |  3,337 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/sha2_bench-ae5b4911aa6bad3725938e02b6a4b0612156091d.md) | 9,906 |  14,793,960 |  1,442 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/regex-ae5b4911aa6bad3725938e02b6a4b0612156091d.md) | 1,387 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/ecrecover-ae5b4911aa6bad3725938e02b6a4b0612156091d.md) | 601 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/pairing-ae5b4911aa6bad3725938e02b6a4b0612156091d.md) | 881 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/kitchen_sink-ae5b4911aa6bad3725938e02b6a4b0612156091d.md) | 3,825 |  2,579,903 |  944 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ae5b4911aa6bad3725938e02b6a4b0612156091d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27214274702)
