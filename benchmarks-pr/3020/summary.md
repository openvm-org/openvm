| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-bc3e4089a4bebd8484bbe5a22b6f2958322b11e7.md) | 564 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-bc3e4089a4bebd8484bbe5a22b6f2958322b11e7.md) | 7,445 |  14,365,133 |  1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-bc3e4089a4bebd8484bbe5a22b6f2958322b11e7.md) | 4,461 |  11,167,961 |  531 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-bc3e4089a4bebd8484bbe5a22b6f2958322b11e7.md) | 604 |  4,090,656 |  198 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-bc3e4089a4bebd8484bbe5a22b6f2958322b11e7.md) | 221 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-bc3e4089a4bebd8484bbe5a22b6f2958322b11e7.md) | 252 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-bc3e4089a4bebd8484bbe5a22b6f2958322b11e7.md) | 2,752 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bc3e4089a4bebd8484bbe5a22b6f2958322b11e7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29394035313)
