| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/fibonacci-a330369cc8d8821e937dc07bdc53852d75714aa0.md) | 471 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/keccak-a330369cc8d8821e937dc07bdc53852d75714aa0.md) | 8,814 |  14,365,133 |  1,541 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/sha2_bench-a330369cc8d8821e937dc07bdc53852d75714aa0.md) | 3,974 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/regex-a330369cc8d8821e937dc07bdc53852d75714aa0.md) | 504 |  4,090,656 |  190 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/ecrecover-a330369cc8d8821e937dc07bdc53852d75714aa0.md) | 217 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/pairing-a330369cc8d8821e937dc07bdc53852d75714aa0.md) | 266 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/kitchen_sink-a330369cc8d8821e937dc07bdc53852d75714aa0.md) | 1,924 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a330369cc8d8821e937dc07bdc53852d75714aa0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29366035513)
