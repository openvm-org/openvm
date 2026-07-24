| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/fibonacci-2c6b527e84a8cf5711350dda2f2904fa95dddfe4.md) |<span style='color: green'>(-8 [-0.5%])</span> 1,578 |  12,000,265 | <span style='color: red'>(+1 [+0.3%])</span> 362 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/keccak-2c6b527e84a8cf5711350dda2f2904fa95dddfe4.md) |<span style='color: red'>(+92 [+1.0%])</span> 9,347 |  18,655,329 | <span style='color: red'>(+9 [+0.6%])</span> 1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/sha2_bench-2c6b527e84a8cf5711350dda2f2904fa95dddfe4.md) |<span style='color: red'>(+55 [+1.1%])</span> 4,930 |  14,793,960 | <span style='color: red'>(+5 [+0.9%])</span> 577 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/regex-2c6b527e84a8cf5711350dda2f2904fa95dddfe4.md) |<span style='color: green'>(-9 [-1.4%])</span> 653 |  4,137,067 | <span style='color: red'>(+4 [+1.9%])</span> 214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/ecrecover-2c6b527e84a8cf5711350dda2f2904fa95dddfe4.md) |<span style='color: red'>(+11 [+2.6%])</span> 438 |  123,583 | <span style='color: green'>(-3 [-1.6%])</span> 182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/pairing-2c6b527e84a8cf5711350dda2f2904fa95dddfe4.md) |<span style='color: red'>(+19 [+3.3%])</span> 589 |  1,745,757 | <span style='color: green'>(-1 [-0.5%])</span> 191 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/kitchen_sink-2c6b527e84a8cf5711350dda2f2904fa95dddfe4.md) |<span style='color: green'>(-6 [-0.3%])</span> 2,208 |  2,579,903 | <span style='color: green'>(-2 [-0.4%])</span> 475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2c6b527e84a8cf5711350dda2f2904fa95dddfe4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30106232428)
