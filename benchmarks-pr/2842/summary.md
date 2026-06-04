| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2842/fibonacci-c822329b0ba256f215c9f517ce0039d23facf258.md) | 3,784 |  12,000,265 |  920 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2842/keccak-c822329b0ba256f215c9f517ce0039d23facf258.md) | 18,741 |  18,655,329 |  3,332 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2842/sha2_bench-c822329b0ba256f215c9f517ce0039d23facf258.md) | 10,219 |  14,793,960 |  1,456 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2842/regex-c822329b0ba256f215c9f517ce0039d23facf258.md) | 1,408 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2842/ecrecover-c822329b0ba256f215c9f517ce0039d23facf258.md) | 602 |  123,583 |  257 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2842/pairing-c822329b0ba256f215c9f517ce0039d23facf258.md) | 890 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2842/kitchen_sink-c822329b0ba256f215c9f517ce0039d23facf258.md) | 1,906 |  2,579,903 |  416 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c822329b0ba256f215c9f517ce0039d23facf258

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26971984614)
