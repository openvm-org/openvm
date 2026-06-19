| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-79ab4929d637bccfd9489d4696656b29277760ec.md) | 3,076 |  12,000,265 | <span style='color: green'>(-3813 [-85.0%])</span> 673 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-79ab4929d637bccfd9489d4696656b29277760ec.md) | 16,391 |  18,655,329 |  3,048 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-79ab4929d637bccfd9489d4696656b29277760ec.md) | 9,298 |  14,793,960 |  1,140 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-79ab4929d637bccfd9489d4696656b29277760ec.md) | 1,181 |  4,137,067 | <span style='color: green'>(-11639 [-97.0%])</span> 358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-79ab4929d637bccfd9489d4696656b29277760ec.md) | 606 |  123,583 | <span style='color: green'>(-5572 [-95.2%])</span> 284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-79ab4929d637bccfd9489d4696656b29277760ec.md) | 948 |  1,745,757 | <span style='color: green'>(-6079 [-95.3%])</span> 301 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-79ab4929d637bccfd9489d4696656b29277760ec.md) | 4,085 |  2,579,903 |  875 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-79ab4929d637bccfd9489d4696656b29277760ec.md) | 1,388 |  12,000,265 |  286 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-79ab4929d637bccfd9489d4696656b29277760ec.md) | 643 |  4,137,067 |  165 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-79ab4929d637bccfd9489d4696656b29277760ec.md) | 390 |  123,583 |  142 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-79ab4929d637bccfd9489d4696656b29277760ec.md) | 526 |  1,745,757 |  149 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-79ab4929d637bccfd9489d4696656b29277760ec.md) | 1,992 |  2,579,903 |  383 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/79ab4929d637bccfd9489d4696656b29277760ec

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27844673939)
