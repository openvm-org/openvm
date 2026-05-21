| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/fibonacci-337518bc756ad3f5a18a036693705c12112fc428.md) | 1,570 |  4,000,051 |  435 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/keccak-337518bc756ad3f5a18a036693705c12112fc428.md) | 13,828 |  14,365,133 |  2,190 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/sha2_bench-337518bc756ad3f5a18a036693705c12112fc428.md) | 9,158 |  11,167,961 |  1,413 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/regex-337518bc756ad3f5a18a036693705c12112fc428.md) | 1,473 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/ecrecover-337518bc756ad3f5a18a036693705c12112fc428.md) | 474 |  112,210 |  263 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/pairing-337518bc756ad3f5a18a036693705c12112fc428.md) | 605 |  592,827 |  256 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/kitchen_sink-337518bc756ad3f5a18a036693705c12112fc428.md) | 2,223 |  1,979,971 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/337518bc756ad3f5a18a036693705c12112fc428

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26239025082)
