| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/fibonacci-02d5b92e5347336c01db10350584c72a1197827b.md) | 3,861 |  12,000,265 |  943 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/keccak-02d5b92e5347336c01db10350584c72a1197827b.md) | 15,768 |  1,235,218 |  2,203 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/regex-02d5b92e5347336c01db10350584c72a1197827b.md) | 1,416 |  4,136,694 |  368 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/ecrecover-02d5b92e5347336c01db10350584c72a1197827b.md) | 639 |  122,348 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/pairing-02d5b92e5347336c01db10350584c72a1197827b.md) | 919 |  1,745,757 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/kitchen_sink-02d5b92e5347336c01db10350584c72a1197827b.md) | 2,382 |  154,763 |  418 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/02d5b92e5347336c01db10350584c72a1197827b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23864342533)
