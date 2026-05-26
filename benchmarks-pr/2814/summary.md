| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/fibonacci-f19a38acdb0d05bc4a8f917905c6340ac7aa980f.md) | 3,756 |  12,000,265 |  920 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/keccak-f19a38acdb0d05bc4a8f917905c6340ac7aa980f.md) | 18,538 |  18,655,329 |  3,263 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/sha2_bench-f19a38acdb0d05bc4a8f917905c6340ac7aa980f.md) | 10,197 |  14,793,960 |  1,463 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/regex-f19a38acdb0d05bc4a8f917905c6340ac7aa980f.md) | 1,385 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/ecrecover-f19a38acdb0d05bc4a8f917905c6340ac7aa980f.md) | 597 |  123,583 |  246 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/pairing-f19a38acdb0d05bc4a8f917905c6340ac7aa980f.md) | 892 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/kitchen_sink-f19a38acdb0d05bc4a8f917905c6340ac7aa980f.md) | 1,900 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f19a38acdb0d05bc4a8f917905c6340ac7aa980f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26462588463)
