| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-3008ad9163887c3e99903b05f4121a72736f381a.md) | 852 |  4,000,051 |  390 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-3008ad9163887c3e99903b05f4121a72736f381a.md) | 15,109 |  14,365,133 |  3,030 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-3008ad9163887c3e99903b05f4121a72736f381a.md) | 7,708 |  11,167,961 |  1,010 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-3008ad9163887c3e99903b05f4121a72736f381a.md) | 875 |  4,090,656 |  287 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-3008ad9163887c3e99903b05f4121a72736f381a.md) | 301 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-3008ad9163887c3e99903b05f4121a72736f381a.md) | 407 |  592,827 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-3008ad9163887c3e99903b05f4121a72736f381a.md) | 3,626 |  1,979,971 |  856 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3008ad9163887c3e99903b05f4121a72736f381a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29054597076)
