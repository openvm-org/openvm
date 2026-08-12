| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-126c180750575c04fd900130bc3e46b94204cd5f.md) | 464 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-126c180750575c04fd900130bc3e46b94204cd5f.md) | 7,373 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-126c180750575c04fd900130bc3e46b94204cd5f.md) | 4,175 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-126c180750575c04fd900130bc3e46b94204cd5f.md) | 669 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-126c180750575c04fd900130bc3e46b94204cd5f.md) | 197 |  112,210 |  198 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-126c180750575c04fd900130bc3e46b94204cd5f.md) | 235 |  592,827 |  195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-126c180750575c04fd900130bc3e46b94204cd5f.md) | 2,041 |  1,979,971 |  527 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/126c180750575c04fd900130bc3e46b94204cd5f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31611690752)
