| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/fibonacci-55d6f46a02945ce7ab6b4527fdaa135bd27afdf6.md) | 3,738 |  12,000,265 |  908 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/keccak-55d6f46a02945ce7ab6b4527fdaa135bd27afdf6.md) | 18,296 |  18,655,329 |  3,208 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/sha2_bench-55d6f46a02945ce7ab6b4527fdaa135bd27afdf6.md) | 10,132 |  14,793,960 |  1,445 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/regex-55d6f46a02945ce7ab6b4527fdaa135bd27afdf6.md) | 1,402 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/ecrecover-55d6f46a02945ce7ab6b4527fdaa135bd27afdf6.md) | 605 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/pairing-55d6f46a02945ce7ab6b4527fdaa135bd27afdf6.md) | 890 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/kitchen_sink-55d6f46a02945ce7ab6b4527fdaa135bd27afdf6.md) | 1,893 |  2,579,903 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/55d6f46a02945ce7ab6b4527fdaa135bd27afdf6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26307917329)
