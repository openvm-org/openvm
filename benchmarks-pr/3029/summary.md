| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/fibonacci-3ad11954a34ed698960fe9c56681037c90faf649.md) | 1,585 |  12,000,265 | <span style='color: green'>(-1 [-0.3%])</span> 360 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/keccak-3ad11954a34ed698960fe9c56681037c90faf649.md) |<span style='color: green'>(-95 [-1.0%])</span> 9,160 |  18,655,329 | <span style='color: green'>(-12 [-0.8%])</span> 1,503 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/sha2_bench-3ad11954a34ed698960fe9c56681037c90faf649.md) |<span style='color: red'>(+19 [+0.4%])</span> 4,894 |  14,793,960 | <span style='color: red'>(+1 [+0.2%])</span> 573 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/regex-3ad11954a34ed698960fe9c56681037c90faf649.md) |<span style='color: green'>(-8 [-1.2%])</span> 654 |  4,137,067 | <span style='color: red'>(+4 [+1.9%])</span> 214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/ecrecover-3ad11954a34ed698960fe9c56681037c90faf649.md) |<span style='color: green'>(-2 [-0.5%])</span> 425 |  123,583 | <span style='color: green'>(-1 [-0.5%])</span> 184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/pairing-3ad11954a34ed698960fe9c56681037c90faf649.md) |<span style='color: green'>(-19 [-3.3%])</span> 551 |  1,745,757 | <span style='color: green'>(-2 [-1.0%])</span> 190 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/kitchen_sink-3ad11954a34ed698960fe9c56681037c90faf649.md) |<span style='color: red'>(+7 [+0.3%])</span> 2,221 |  2,579,903 | <span style='color: green'>(-1 [-0.2%])</span> 476 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3ad11954a34ed698960fe9c56681037c90faf649

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29612392480)
