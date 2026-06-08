| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/fibonacci-814c620cb787f91e51e46c2efc1eb0c72a2f89a8.md) | 1,412 |  4,000,051 |  434 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/keccak-814c620cb787f91e51e46c2efc1eb0c72a2f89a8.md) | 13,890 |  14,365,133 |  2,396 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/sha2_bench-814c620cb787f91e51e46c2efc1eb0c72a2f89a8.md) | 8,803 |  11,167,961 |  1,400 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/regex-814c620cb787f91e51e46c2efc1eb0c72a2f89a8.md) | 1,377 |  4,090,656 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/ecrecover-814c620cb787f91e51e46c2efc1eb0c72a2f89a8.md) | 435 |  112,210 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/pairing-814c620cb787f91e51e46c2efc1eb0c72a2f89a8.md) | 573 |  592,827 |  256 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/kitchen_sink-814c620cb787f91e51e46c2efc1eb0c72a2f89a8.md) | 3,709 |  1,979,971 |  942 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/814c620cb787f91e51e46c2efc1eb0c72a2f89a8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27150507835)
