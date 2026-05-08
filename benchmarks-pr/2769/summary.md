| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/fibonacci-451837cc053a507ca6afbef06a29a3caaee8f9f3.md) | 3,795 |  12,000,265 |  944 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/keccak-451837cc053a507ca6afbef06a29a3caaee8f9f3.md) | 18,886 |  18,655,329 |  3,366 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/sha2_bench-451837cc053a507ca6afbef06a29a3caaee8f9f3.md) | 9,031 |  14,793,960 |  1,394 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/regex-451837cc053a507ca6afbef06a29a3caaee8f9f3.md) | 1,429 |  4,137,067 |  382 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/ecrecover-451837cc053a507ca6afbef06a29a3caaee8f9f3.md) | 637 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/pairing-451837cc053a507ca6afbef06a29a3caaee8f9f3.md) | 899 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/kitchen_sink-451837cc053a507ca6afbef06a29a3caaee8f9f3.md) | 2,127 |  2,579,903 |  443 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/451837cc053a507ca6afbef06a29a3caaee8f9f3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25564527846)
