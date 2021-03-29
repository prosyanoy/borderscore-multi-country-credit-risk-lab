# BorderScore monitoring report

Reference: fixed holdout. Current: simulated production batch. Thresholds: PSI >= 0.10 warn, >= 0.25 alert.

## Scenario

| country | covariate_shift | label_shift |
|---|---|---|
| BR | 0.000 | 0.600 |
| MX | 2.000 | 0.000 |
| PL | 0.000 | 0.000 |
| ID | 0.800 | 0.000 |


## Country summary (labels assumed matured)

| country | ref_default_rate | cur_default_rate | mean_score_ref | mean_score_cur | auc_cur | ece_cur |
|---|---|---|---|---|---|---|
| BR | 0.079 | 0.130 | 0.088 | 0.089 | 0.799 | 0.041 |
| MX | 0.115 | 0.462 | 0.107 | 0.426 | 0.851 | 0.044 |
| PL | 0.040 | 0.042 | 0.039 | 0.038 | 0.780 | 0.009 |
| ID | 0.136 | 0.271 | 0.131 | 0.252 | 0.826 | 0.024 |


## Score drift

| country | psi | status |
|---|---|---|
| BR | 0.006 | ok |
| MX | 1.941 | alert |
| PL | 0.003 | ok |
| ID | 0.455 | alert |


## Feature drift (flagged only)

| country | item | psi | status |
|---|---|---|---|
| ID | dti | 0.258 | alert |
| ID | log_income | 0.109 | warn |
| ID | utilization | 0.104 | warn |
| MX | dti | 1.372 | alert |
| MX | log_income | 0.770 | alert |
| MX | delinq_2y | 0.486 | alert |
| MX | utilization | 0.481 | alert |
| MX | loan_to_income | 0.380 | alert |


![PSI heatmap](psi_heatmap.png)
