| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-4215bf3567d430f55ac6c125ab2fa5f24ad9aae9.md) | 474 |  4,000,051 |  237 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-4215bf3567d430f55ac6c125ab2fa5f24ad9aae9.md) | 7,257 |  14,365,133 |  1,528 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-4215bf3567d430f55ac6c125ab2fa5f24ad9aae9.md) | 4,777 |  11,167,961 |  532 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-4215bf3567d430f55ac6c125ab2fa5f24ad9aae9.md) | 694 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-4215bf3567d430f55ac6c125ab2fa5f24ad9aae9.md) | 226 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-4215bf3567d430f55ac6c125ab2fa5f24ad9aae9.md) | 273 |  592,827 |  191 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-4215bf3567d430f55ac6c125ab2fa5f24ad9aae9.md) | 2,743 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4215bf3567d430f55ac6c125ab2fa5f24ad9aae9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29749426039)
