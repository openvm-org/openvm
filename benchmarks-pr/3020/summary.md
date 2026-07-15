| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-f21f22a23a9c079be3d014cbb9f48df4d96855a9.md) | 565 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-f21f22a23a9c079be3d014cbb9f48df4d96855a9.md) | 7,425 |  14,365,133 |  1,526 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-f21f22a23a9c079be3d014cbb9f48df4d96855a9.md) | 4,521 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-f21f22a23a9c079be3d014cbb9f48df4d96855a9.md) | 596 |  4,090,656 |  190 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-f21f22a23a9c079be3d014cbb9f48df4d96855a9.md) | 225 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-f21f22a23a9c079be3d014cbb9f48df4d96855a9.md) | 250 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-f21f22a23a9c079be3d014cbb9f48df4d96855a9.md) | 2,758 |  1,979,971 |  474 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f21f22a23a9c079be3d014cbb9f48df4d96855a9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29391095700)
