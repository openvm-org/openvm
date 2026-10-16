| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3169/fibonacci-a3b094bc3d48ddca6de8e62812fbfe2f9d1dcce4.md) |<span style='color: green'>(-23 [-1.4%])</span> 1,654 |  12,000,265 |  367 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3169/keccak-a3b094bc3d48ddca6de8e62812fbfe2f9d1dcce4.md) |<span style='color: green'>(-26 [-0.3%])</span> 9,667 |  18,655,329 | <span style='color: green'>(-3 [-0.2%])</span> 1,544 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3169/sha2_bench-a3b094bc3d48ddca6de8e62812fbfe2f9d1dcce4.md) |<span style='color: red'>(+86 [+1.6%])</span> 5,419 |  14,793,960 | <span style='color: green'>(-4 [-0.7%])</span> 592 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3169/regex-a3b094bc3d48ddca6de8e62812fbfe2f9d1dcce4.md) |<span style='color: green'>(-5 [-0.7%])</span> 699 |  4,137,067 | <span style='color: red'>(+1 [+0.5%])</span> 218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3169/ecrecover-a3b094bc3d48ddca6de8e62812fbfe2f9d1dcce4.md) |<span style='color: red'>(+7 [+1.6%])</span> 437 |  123,583 | <span style='color: green'>(-1 [-0.5%])</span> 188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3169/pairing-a3b094bc3d48ddca6de8e62812fbfe2f9d1dcce4.md) |<span style='color: red'>(+48 [+8.5%])</span> 611 |  1,745,757 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3169/kitchen_sink-a3b094bc3d48ddca6de8e62812fbfe2f9d1dcce4.md) | 2,308 |  2,579,903 |  493 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a3b094bc3d48ddca6de8e62812fbfe2f9d1dcce4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/38979868193)
