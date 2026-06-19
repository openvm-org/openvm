| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2908/fibonacci-87700d72e28c8cc758d9d76eada7bf8d5bd322e5.md) | 1,389 |  4,000,051 |  388 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2908/keccak-87700d72e28c8cc758d9d76eada7bf8d5bd322e5.md) | 16,180 |  14,365,133 |  3,023 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2908/sha2_bench-87700d72e28c8cc758d9d76eada7bf8d5bd322e5.md) | 10,109 |  11,167,961 |  1,018 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2908/regex-87700d72e28c8cc758d9d76eada7bf8d5bd322e5.md) | 1,587 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2908/ecrecover-87700d72e28c8cc758d9d76eada7bf8d5bd322e5.md) | 446 |  112,210 |  310 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2908/pairing-87700d72e28c8cc758d9d76eada7bf8d5bd322e5.md) | 596 |  592,827 |  298 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2908/kitchen_sink-87700d72e28c8cc758d9d76eada7bf8d5bd322e5.md) | 3,912 |  1,979,971 |  857 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/87700d72e28c8cc758d9d76eada7bf8d5bd322e5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27821591264)
