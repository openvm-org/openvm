| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/fibonacci-64ca1c2367b8988fd86171f8926478b4f42cf3b8.md) | 3,768 |  12,000,265 |  919 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/keccak-64ca1c2367b8988fd86171f8926478b4f42cf3b8.md) | 18,772 |  18,655,329 |  3,321 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/sha2_bench-64ca1c2367b8988fd86171f8926478b4f42cf3b8.md) | 10,195 |  14,793,960 |  1,462 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/regex-64ca1c2367b8988fd86171f8926478b4f42cf3b8.md) | 1,391 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/ecrecover-64ca1c2367b8988fd86171f8926478b4f42cf3b8.md) | 600 |  123,583 |  248 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/pairing-64ca1c2367b8988fd86171f8926478b4f42cf3b8.md) | 894 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/kitchen_sink-64ca1c2367b8988fd86171f8926478b4f42cf3b8.md) | 1,896 |  2,579,903 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/64ca1c2367b8988fd86171f8926478b4f42cf3b8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26472432189)
