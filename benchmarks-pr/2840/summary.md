| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/fibonacci-c0b116528167e733e548394631e32480875a97e1.md) | 1,406 |  4,000,051 |  442 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/keccak-c0b116528167e733e548394631e32480875a97e1.md) | 13,534 |  14,365,133 |  2,357 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/sha2_bench-c0b116528167e733e548394631e32480875a97e1.md) | 9,029 |  11,167,961 |  1,422 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/regex-c0b116528167e733e548394631e32480875a97e1.md) | 1,489 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/ecrecover-c0b116528167e733e548394631e32480875a97e1.md) | 441 |  112,210 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/pairing-c0b116528167e733e548394631e32480875a97e1.md) | 573 |  592,827 |  253 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/kitchen_sink-c0b116528167e733e548394631e32480875a97e1.md) | 3,753 |  1,979,971 |  953 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c0b116528167e733e548394631e32480875a97e1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26963591825)
