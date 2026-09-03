| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/fibonacci-5bdc769320d126793f07740a98edbf95ce3ed8f6.md) | 485 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/keccak-5bdc769320d126793f07740a98edbf95ce3ed8f6.md) | 7,826 |  14,365,133 |  1,646 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/sha2_bench-5bdc769320d126793f07740a98edbf95ce3ed8f6.md) | 4,359 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/regex-5bdc769320d126793f07740a98edbf95ce3ed8f6.md) | 748 |  4,090,656 |  221 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/ecrecover-5bdc769320d126793f07740a98edbf95ce3ed8f6.md) | 210 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/pairing-5bdc769320d126793f07740a98edbf95ce3ed8f6.md) | 250 |  592,827 |  175 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/kitchen_sink-5bdc769320d126793f07740a98edbf95ce3ed8f6.md) | 2,252 |  1,979,971 |  478 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5bdc769320d126793f07740a98edbf95ce3ed8f6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33816986882)
