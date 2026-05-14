| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/fibonacci-2acd66ceb91fa7a766c8152ad7d9e5ebd74f4dec.md) | 3,714 |  12,000,265 |  912 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/keccak-2acd66ceb91fa7a766c8152ad7d9e5ebd74f4dec.md) | 18,455 |  18,655,329 |  3,260 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/sha2_bench-2acd66ceb91fa7a766c8152ad7d9e5ebd74f4dec.md) | 10,091 |  14,793,960 |  1,446 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/regex-2acd66ceb91fa7a766c8152ad7d9e5ebd74f4dec.md) | 1,397 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/ecrecover-2acd66ceb91fa7a766c8152ad7d9e5ebd74f4dec.md) | 602 |  123,583 |  254 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/pairing-2acd66ceb91fa7a766c8152ad7d9e5ebd74f4dec.md) | 885 |  1,745,757 |  258 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/kitchen_sink-2acd66ceb91fa7a766c8152ad7d9e5ebd74f4dec.md) | 1,888 |  2,579,903 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2acd66ceb91fa7a766c8152ad7d9e5ebd74f4dec

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25887727732)
