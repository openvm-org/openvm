| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3133/fibonacci-122958c4026b74e9b0925e08002e69b570152b37.md) |<span style='color: green'>(-4 [-0.2%])</span> 1,680 |  12,000,265 | <span style='color: green'>(-2 [-0.5%])</span> 368 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3133/keccak-122958c4026b74e9b0925e08002e69b570152b37.md) |<span style='color: red'>(+67 [+0.7%])</span> 9,585 |  18,655,329 | <span style='color: green'>(-17 [-1.1%])</span> 1,547 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3133/sha2_bench-122958c4026b74e9b0925e08002e69b570152b37.md) | 5,358 |  14,793,960 | <span style='color: red'>(+14 [+2.4%])</span> 600 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3133/regex-122958c4026b74e9b0925e08002e69b570152b37.md) |<span style='color: green'>(-5 [-0.7%])</span> 692 |  4,137,067 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3133/ecrecover-122958c4026b74e9b0925e08002e69b570152b37.md) |<span style='color: green'>(-1 [-0.2%])</span> 429 |  123,583 | <span style='color: green'>(-2 [-1.1%])</span> 187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3133/pairing-122958c4026b74e9b0925e08002e69b570152b37.md) |<span style='color: green'>(-22 [-3.7%])</span> 568 |  1,745,757 | <span style='color: red'>(+1 [+0.5%])</span> 194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3133/kitchen_sink-122958c4026b74e9b0925e08002e69b570152b37.md) |<span style='color: green'>(-3 [-0.1%])</span> 2,297 |  2,579,903 | <span style='color: red'>(+2 [+0.4%])</span> 494 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/122958c4026b74e9b0925e08002e69b570152b37

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33081761853)
