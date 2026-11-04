| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/fibonacci-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) |<span style='color: red'>(+28 [+1.7%])</span> 1,690 |  12,000,265 | <span style='color: red'>(+1 [+0.3%])</span> 370 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/keccak-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) |<span style='color: green'>(-30 [-0.3%])</span> 9,557 |  18,655,329 | <span style='color: red'>(+10 [+0.7%])</span> 1,548 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/sha2_bench-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) |<span style='color: red'>(+57 [+1.1%])</span> 5,299 |  14,793,960 |  587 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/regex-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) |<span style='color: red'>(+10 [+1.5%])</span> 697 |  4,137,067 | <span style='color: red'>(+1 [+0.5%])</span> 222 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/ecrecover-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) |<span style='color: red'>(+10 [+2.3%])</span> 440 |  123,583 | <span style='color: red'>(+2 [+1.1%])</span> 189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/pairing-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) |<span style='color: red'>(+5 [+0.9%])</span> 589 |  1,745,757 | <span style='color: red'>(+3 [+1.5%])</span> 197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/kitchen_sink-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) |<span style='color: red'>(+18 [+0.8%])</span> 2,317 |  2,579,903 | <span style='color: red'>(+5 [+1.0%])</span> 497 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/05f3b73e7294fbb6d38155abafc98f1fe2bcfb73

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/41863319488)
