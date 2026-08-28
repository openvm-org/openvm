| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/fibonacci-f2bf9b662d81e9e022b08e42ff3c46441cccd527.md) | 481 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/keccak-f2bf9b662d81e9e022b08e42ff3c46441cccd527.md) | 7,542 |  14,365,133 |  1,615 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/sha2_bench-f2bf9b662d81e9e022b08e42ff3c46441cccd527.md) | 4,395 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/regex-f2bf9b662d81e9e022b08e42ff3c46441cccd527.md) | 767 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/ecrecover-f2bf9b662d81e9e022b08e42ff3c46441cccd527.md) | 207 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/pairing-f2bf9b662d81e9e022b08e42ff3c46441cccd527.md) | 251 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/kitchen_sink-f2bf9b662d81e9e022b08e42ff3c46441cccd527.md) | 2,228 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f2bf9b662d81e9e022b08e42ff3c46441cccd527

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33189154591)
