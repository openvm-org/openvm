| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3000/fibonacci-27bc1868d9286b5070a1d1959e0e6f91faf646ce.md) |<span style='color: red'>(+67 [+2.2%])</span> 3,069 |  12,000,265 | <span style='color: red'>(+17 [+2.6%])</span> 676 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3000/keccak-27bc1868d9286b5070a1d1959e0e6f91faf646ce.md) | 16,397 |  18,655,329 |  3,037 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3000/sha2_bench-27bc1868d9286b5070a1d1959e0e6f91faf646ce.md) |<span style='color: red'>(+92 [+1.0%])</span> 9,336 |  14,793,960 | <span style='color: red'>(+16 [+1.4%])</span> 1,143 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3000/regex-27bc1868d9286b5070a1d1959e0e6f91faf646ce.md) |<span style='color: red'>(+5 [+0.4%])</span> 1,166 |  4,137,067 | <span style='color: red'>(+5 [+1.4%])</span> 357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3000/ecrecover-27bc1868d9286b5070a1d1959e0e6f91faf646ce.md) |<span style='color: red'>(+11 [+1.9%])</span> 605 |  123,583 | <span style='color: red'>(+1 [+0.4%])</span> 285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3000/pairing-27bc1868d9286b5070a1d1959e0e6f91faf646ce.md) |<span style='color: red'>(+15 [+1.6%])</span> 956 |  1,745,757 | <span style='color: green'>(-4 [-1.3%])</span> 304 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3000/kitchen_sink-27bc1868d9286b5070a1d1959e0e6f91faf646ce.md) |<span style='color: green'>(-64 [-1.5%])</span> 4,131 |  2,579,903 | <span style='color: green'>(-23 [-2.5%])</span> 881 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/27bc1868d9286b5070a1d1959e0e6f91faf646ce

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29069975114)
