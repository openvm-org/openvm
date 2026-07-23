| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-ba807f2e91774734b514f421322e16b65669498d.md) | 480 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-ba807f2e91774734b514f421322e16b65669498d.md) | 7,342 |  14,365,133 |  1,537 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-ba807f2e91774734b514f421322e16b65669498d.md) | 4,809 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-ba807f2e91774734b514f421322e16b65669498d.md) | 680 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-ba807f2e91774734b514f421322e16b65669498d.md) | 231 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-ba807f2e91774734b514f421322e16b65669498d.md) | 276 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-ba807f2e91774734b514f421322e16b65669498d.md) | 2,755 |  1,979,971 |  467 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ba807f2e91774734b514f421322e16b65669498d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30033825411)
