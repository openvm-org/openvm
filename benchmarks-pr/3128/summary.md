| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3128/fibonacci-52948d1a57ad4122604baa48087ed9496135d244.md) | 1,695 |  12,000,265 |  370 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3128/keccak-52948d1a57ad4122604baa48087ed9496135d244.md) | 9,575 |  18,655,329 |  1,536 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3128/sha2_bench-52948d1a57ad4122604baa48087ed9496135d244.md) | 5,240 |  14,793,960 |  587 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3128/regex-52948d1a57ad4122604baa48087ed9496135d244.md) | 699 |  4,137,067 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3128/ecrecover-52948d1a57ad4122604baa48087ed9496135d244.md) | 439 |  123,583 |  192 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3128/pairing-52948d1a57ad4122604baa48087ed9496135d244.md) | 600 |  1,745,757 |  198 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3128/kitchen_sink-52948d1a57ad4122604baa48087ed9496135d244.md) | 2,308 |  2,579,903 |  496 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/52948d1a57ad4122604baa48087ed9496135d244

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32520792443)
