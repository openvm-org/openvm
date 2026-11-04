| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/fibonacci-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) |<span style='color: red'>(+16 [+1.0%])</span> 1,678 |  12,000,265 | <span style='color: green'>(-1 [-0.3%])</span> 368 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/keccak-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) |<span style='color: red'>(+10 [+0.1%])</span> 9,597 |  18,655,329 | <span style='color: red'>(+2 [+0.1%])</span> 1,540 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/sha2_bench-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) |<span style='color: red'>(+105 [+2.0%])</span> 5,347 |  14,793,960 | <span style='color: red'>(+3 [+0.5%])</span> 590 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/regex-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) |<span style='color: red'>(+15 [+2.2%])</span> 702 |  4,137,067 | <span style='color: green'>(-5 [-2.3%])</span> 216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/ecrecover-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) |<span style='color: red'>(+10 [+2.3%])</span> 440 |  123,583 | <span style='color: red'>(+3 [+1.6%])</span> 190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/pairing-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) |<span style='color: red'>(+24 [+4.1%])</span> 608 |  1,745,757 | <span style='color: red'>(+1 [+0.5%])</span> 195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/kitchen_sink-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) | 2,297 |  2,579,903 | <span style='color: green'>(-1 [-0.2%])</span> 491 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/fibonacci_e2e-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) | 1,714 |  12,000,265 |  356 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/regex_e2e-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) | 854 |  4,137,067 |  212 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/ecrecover_e2e-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) | 500 |  123,583 |  178 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/pairing_e2e-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) | 676 |  1,745,757 |  185 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3191/kitchen_sink_e2e-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) | 2,548 |  2,579,903 |  487 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/05f3b73e7294fbb6d38155abafc98f1fe2bcfb73

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/41863611551)
