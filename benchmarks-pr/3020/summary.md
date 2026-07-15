| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-d0d1dafe5e42ac6cea002336f46915ed4006a34d.md) | 564 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-d0d1dafe5e42ac6cea002336f46915ed4006a34d.md) | 7,502 |  14,365,133 |  1,536 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-d0d1dafe5e42ac6cea002336f46915ed4006a34d.md) | 4,492 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-d0d1dafe5e42ac6cea002336f46915ed4006a34d.md) | 594 |  4,090,656 |  194 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-d0d1dafe5e42ac6cea002336f46915ed4006a34d.md) | 224 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-d0d1dafe5e42ac6cea002336f46915ed4006a34d.md) | 247 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-d0d1dafe5e42ac6cea002336f46915ed4006a34d.md) | 2,673 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d0d1dafe5e42ac6cea002336f46915ed4006a34d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29392885156)
