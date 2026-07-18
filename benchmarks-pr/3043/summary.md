| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/fibonacci-f03af18cbfaaa7c23fa8fb15e007b07e4a0553e9.md) | 417 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/keccak-f03af18cbfaaa7c23fa8fb15e007b07e4a0553e9.md) | 8,675 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/sha2_bench-f03af18cbfaaa7c23fa8fb15e007b07e4a0553e9.md) | 4,203 |  11,167,961 |  514 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/regex-f03af18cbfaaa7c23fa8fb15e007b07e4a0553e9.md) | 575 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/ecrecover-f03af18cbfaaa7c23fa8fb15e007b07e4a0553e9.md) | 218 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/pairing-f03af18cbfaaa7c23fa8fb15e007b07e4a0553e9.md) | 282 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/kitchen_sink-f03af18cbfaaa7c23fa8fb15e007b07e4a0553e9.md) | 1,917 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f03af18cbfaaa7c23fa8fb15e007b07e4a0553e9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29658421340)
