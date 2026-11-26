# Loyaltics fintech challenger

COFINFAD (Colombia, 2023), DOI 10.17632/mhb4zn3258.1, CC BY 4.0. Source attribution: Luis Eduardo Muñoz Guerrero, Yony Fernando Ceballos, Luis David Trejos Rojas (2025). Author's pinned Hugging Face CSV mirror. This is not South African banking data. No raw transaction data is committed or uploaded as a workflow artifact.

Run `python train_fintech.py` with the dependencies in the fintech workflow. The script uses dated nonzero transactions only and excludes full-year customer fields, supplied churn scores, and supplied CLV. It models 30-day future transaction inactivity and conditional transaction turnover, not actual bank revenue or closure. Chronological and customer-disjoint partitions, separate calibration, twelve candidates, bootstrapped PR-AUC, activity-based subgroup checks and feature-family stress tests are included.

The workflow produces a separate candidate bundle and report. The live UK model and nine-input scoring API are unchanged until the candidate is validated and its richer transaction-input schema and new financial semantics are integrated. The release gate is a minimum baseline check, not proof of local validity or fairness. Financial Diaries and Findex are not merged with unrelated customer records. Do not use restricted Nedbank competition data without permission.
