| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/fibonacci-8a8f22bfcc4b95cc43d38e3cfe973d6aa6f40c92.md) | 476 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/keccak-8a8f22bfcc4b95cc43d38e3cfe973d6aa6f40c92.md) | 7,366 |  14,365,133 |  1,527 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/sha2_bench-8a8f22bfcc4b95cc43d38e3cfe973d6aa6f40c92.md) | 4,141 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/regex-8a8f22bfcc4b95cc43d38e3cfe973d6aa6f40c92.md) | 649 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/ecrecover-8a8f22bfcc4b95cc43d38e3cfe973d6aa6f40c92.md) | 228 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/pairing-8a8f22bfcc4b95cc43d38e3cfe973d6aa6f40c92.md) | 240 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/kitchen_sink-8a8f22bfcc4b95cc43d38e3cfe973d6aa6f40c92.md) | 2,047 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8a8f22bfcc4b95cc43d38e3cfe973d6aa6f40c92

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30840890967)
