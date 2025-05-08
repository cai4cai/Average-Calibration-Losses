
=== Results for dataset: AMOS ===

| Metric    |   Baseline vs HardL1-ACE p-value |   Baseline vs SoftL1-ACE p-value |
|:----------|---------------------------------:|---------------------------------:|
| macro_ACE |                        0.01148   |                        3.839e-22 |
| macro_ECE |                        0.2244    |                        0.08832   |
| macro_MCE |                        0.0009338 |                        1.932e-28 |
| DSC       |                        0.02925   |                        1.13e-07  |

=== Results for dataset: ACDC ===

| Metric    |   Baseline vs HardL1-ACE p-value |   Baseline vs SoftL1-ACE p-value |
|:----------|---------------------------------:|---------------------------------:|
| macro_ACE |                        6.991e-26 |                        1.385e-42 |
| macro_ECE |                        3.08e-23  |                        2.092e-26 |
| macro_MCE |                        2.603e-28 |                        1.348e-48 |
| DSC       |                        0.981     |                        1.148e-06 |

=== Results for dataset: KITS ===

| Metric    |   Baseline vs HardL1-ACE p-value |   Baseline vs SoftL1-ACE p-value |
|:----------|---------------------------------:|---------------------------------:|
| macro_ACE |                        4.153e-05 |                        3.962e-11 |
| macro_ECE |                        0.2911    |                        0.7587    |
| macro_MCE |                        8.575e-08 |                        1.479e-14 |
| DSC       |                        0.6471    |                        0.8972    |

=== Results for dataset: BRATS ===

| Metric    |   Baseline vs HardL1-ACE p-value |   Baseline vs SoftL1-ACE p-value |
|:----------|---------------------------------:|---------------------------------:|
| macro_ACE |                        8.048e-05 |                        1.273e-26 |
| macro_ECE |                        0.6345    |                        0.5674    |
| macro_MCE |                        4.976e-06 |                        9.942e-32 |
| DSC       |                        0.8557    |                        1.773e-08 |

Analysis:

For AMOS, “macro_ACE,” “macro_MCE,” and “DSC” show statistically significant differences between the Baseline and both HardL1‐ACE and SoftL1‐ACE (p < .05). The p‐values for “macro_ECE” are not significant, suggesting no strong difference for that metric. SoftL1‐ACE has extremely small p‐values for several metrics, indicating a stronger difference.

For ACDC, “macro_ACE,” “macro_ECE,” and “macro_MCE” appear significantly different when comparing Baseline to both variants (p < .05). DSC shows no significant difference between Baseline and HardL1‐ACE, but it is significant between Baseline and SoftL1‐ACE.

For KiTS, “macro_ACE” and “macro_MCE” appear to have significant differences (p < .05). By contrast, “macro_ECE” and DSC do not show significant differences.

For BraTS, “macro_ACE” and “macro_MCE” are notably significant (p < .05) between Baseline and both HardL1‐ACE and SoftL1‐ACE, while “macro_ECE” is not. DSC is not significant for HardL1‐ACE, but there is a significant difference for SoftL1‐ACE vs. Baseline.

In short, HardL1‐ACE and SoftL1‐ACE often differ significantly from Baseline on certain calibration metrics (e.g., macro_ACE, macro_MCE), but the degree of difference can vary by dataset and metric, with DSC sometimes unaffected or only significant for one variant.