| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/fibonacci-9a3f170530bfebd1fb9065d1b65c52bd6d058178.md) | 3,818 |  12,000,265 |  934 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/keccak-9a3f170530bfebd1fb9065d1b65c52bd6d058178.md) | 18,432 |  18,655,329 |  3,250 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/sha2_bench-9a3f170530bfebd1fb9065d1b65c52bd6d058178.md) | 10,148 |  14,793,960 |  1,454 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/regex-9a3f170530bfebd1fb9065d1b65c52bd6d058178.md) | 1,397 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/ecrecover-9a3f170530bfebd1fb9065d1b65c52bd6d058178.md) | 609 |  123,583 |  254 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/pairing-9a3f170530bfebd1fb9065d1b65c52bd6d058178.md) | 892 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/kitchen_sink-9a3f170530bfebd1fb9065d1b65c52bd6d058178.md) | 1,895 |  2,579,903 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9a3f170530bfebd1fb9065d1b65c52bd6d058178

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26297645968)
