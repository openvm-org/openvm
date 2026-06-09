| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/fibonacci-c77b71bcf572203de3e6b4ecf0490f5cbf0f7252.md) | 5,235 |  4,000,051 |  432 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/keccak-c77b71bcf572203de3e6b4ecf0490f5cbf0f7252.md) | 18,434 |  14,365,133 |  2,362 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/sha2_bench-c77b71bcf572203de3e6b4ecf0490f5cbf0f7252.md) | 12,601 |  11,167,961 |  1,425 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/regex-c77b71bcf572203de3e6b4ecf0490f5cbf0f7252.md) | 3,641 |  4,090,656 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/ecrecover-c77b71bcf572203de3e6b4ecf0490f5cbf0f7252.md) | 1,959 |  112,210 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/pairing-c77b71bcf572203de3e6b4ecf0490f5cbf0f7252.md) | 2,101 |  592,827 |  257 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/kitchen_sink-c77b71bcf572203de3e6b4ecf0490f5cbf0f7252.md) | 6,012 |  1,979,971 |  939 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c77b71bcf572203de3e6b4ecf0490f5cbf0f7252

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27223522718)
