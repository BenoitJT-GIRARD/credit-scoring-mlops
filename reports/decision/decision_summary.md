<!-- Written by scripts/decision_analysis.py on 2026-09-16. Edits here are overwritten. -->

| Question | Answer |
|---|---|
| Cost per applicant at the shipped threshold | 0.4888 (95% CI 0.4721–0.5068) |
| Optimism of tuning the threshold on its own data | 0.0029 per applicant |
| Regret of the shipped threshold vs this sample's optimum | 0.0028 (threshold 0.49 vs 0.48) |

| Baseline | Cost per applicant |
|---|---|
| accept-all | 0.8075 |
| reject-all | 0.9193 |
| random-at-base-rate | 0.8170 |

| Group | n | Default rate | Refusal rate | FNR | Cost |
|---|---|---|---|---|---|
| <30 | 4349 | 0.112 | 0.484 | 0.211 | 0.6326 |
| 30-39 | 8144 | 0.093 | 0.340 | 0.254 | 0.5070 |
| 40-49 | 7773 | 0.078 | 0.258 | 0.327 | 0.4612 |
| 50-59 | 6844 | 0.065 | 0.194 | 0.457 | 0.4541 |
| 60+ | 3641 | 0.051 | 0.125 | 0.587 | 0.4004 |
| F | 20380 | 0.069 | 0.232 | 0.386 | 0.4578 |
| M | 10371 | 0.103 | 0.378 | 0.243 | 0.5498 |

> **How to read it.** One row per group. Default rate is what the group did, counted and
> never predicted, and every column beside it is read against that. Refusal rate says how
> often the shipped policy said no to the group. FNR (false negative rate) says how many of
> its defaulters slipped past. Cost puts both mistakes on a single scale, under the
> ten-to-one ratio this analysis assumes.
