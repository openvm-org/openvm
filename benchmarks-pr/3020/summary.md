| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-8708ff35026a88a1f367bafd800984dff6f29ffc.md) | 474 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-8708ff35026a88a1f367bafd800984dff6f29ffc.md) | 7,127 |  14,365,133 |  1,514 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-8708ff35026a88a1f367bafd800984dff6f29ffc.md) | 4,443 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-8708ff35026a88a1f367bafd800984dff6f29ffc.md) | 672 |  4,090,656 |  222 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-8708ff35026a88a1f367bafd800984dff6f29ffc.md) | 222 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-8708ff35026a88a1f367bafd800984dff6f29ffc.md) | 246 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-8708ff35026a88a1f367bafd800984dff6f29ffc.md) | 2,719 |  1,979,971 |  475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8708ff35026a88a1f367bafd800984dff6f29ffc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29467996874)
