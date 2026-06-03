| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/fibonacci-bee52880863560f7f3843577090c6e84c8505ac4.md) | 3,746 |  12,000,265 |  908 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/keccak-bee52880863560f7f3843577090c6e84c8505ac4.md) | 18,674 |  18,655,329 |  3,290 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/sha2_bench-bee52880863560f7f3843577090c6e84c8505ac4.md) | 10,142 |  14,793,960 |  1,446 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/regex-bee52880863560f7f3843577090c6e84c8505ac4.md) | 1,400 |  4,137,067 |  364 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/ecrecover-bee52880863560f7f3843577090c6e84c8505ac4.md) | 598 |  123,583 |  254 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/pairing-bee52880863560f7f3843577090c6e84c8505ac4.md) | 886 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/kitchen_sink-bee52880863560f7f3843577090c6e84c8505ac4.md) | 1,896 |  2,579,903 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bee52880863560f7f3843577090c6e84c8505ac4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26889722871)
