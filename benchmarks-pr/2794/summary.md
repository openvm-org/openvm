| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/fibonacci-8cf1d0519e22d09dff698f687aacbed9a8c4b5b7.md) | 1,593 |  4,000,051 |  437 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/keccak-8cf1d0519e22d09dff698f687aacbed9a8c4b5b7.md) | 14,012 |  14,365,133 |  2,393 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/sha2_bench-8cf1d0519e22d09dff698f687aacbed9a8c4b5b7.md) | 9,360 |  11,167,961 |  1,430 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/regex-8cf1d0519e22d09dff698f687aacbed9a8c4b5b7.md) | 1,480 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/ecrecover-8cf1d0519e22d09dff698f687aacbed9a8c4b5b7.md) | 481 |  112,210 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/pairing-8cf1d0519e22d09dff698f687aacbed9a8c4b5b7.md) | 592 |  592,827 |  253 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/kitchen_sink-8cf1d0519e22d09dff698f687aacbed9a8c4b5b7.md) | 1,813 |  1,979,971 |  404 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8cf1d0519e22d09dff698f687aacbed9a8c4b5b7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26123340188)
