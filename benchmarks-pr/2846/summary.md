| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-3a4b87abafcd717af9077d6cc195dd3c388cee13.md) | 3,895 |  12,000,265 | <span style='color: green'>(-3355 [-74.8%])</span> 1,131 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-3a4b87abafcd717af9077d6cc195dd3c388cee13.md) | 21,807 |  18,655,329 |  4,618 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-3a4b87abafcd717af9077d6cc195dd3c388cee13.md) | 9,515 |  14,793,960 |  1,820 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-3a4b87abafcd717af9077d6cc195dd3c388cee13.md) | 1,516 |  4,137,067 | <span style='color: green'>(-11568 [-96.4%])</span> 429 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-3a4b87abafcd717af9077d6cc195dd3c388cee13.md) | 606 |  123,583 | <span style='color: green'>(-5570 [-95.1%])</span> 286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-3a4b87abafcd717af9077d6cc195dd3c388cee13.md) | 945 |  1,745,757 | <span style='color: green'>(-6071 [-95.2%])</span> 309 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-3a4b87abafcd717af9077d6cc195dd3c388cee13.md) | 4,145 |  2,579,903 |  878 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-3a4b87abafcd717af9077d6cc195dd3c388cee13.md) | 1,708 |  12,000,265 |  495 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-3a4b87abafcd717af9077d6cc195dd3c388cee13.md) | 718 |  4,137,067 |  197 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-3a4b87abafcd717af9077d6cc195dd3c388cee13.md) | 368 |  123,583 |  143 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-3a4b87abafcd717af9077d6cc195dd3c388cee13.md) | 505 |  1,745,757 |  146 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-3a4b87abafcd717af9077d6cc195dd3c388cee13.md) | 2,175 |  2,579,903 |  383 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3a4b87abafcd717af9077d6cc195dd3c388cee13

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27279609748)
