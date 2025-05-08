Scratch file with results from various things

| Data               | Status         | SegResNet | DiNTS | SwinUNetR | Ref DSC |
|--------------------|----------------|-----------|-------|-----------|---------|
| T01 Brain Tumour   | Done           | 74.7%     | 67.9% | 74.4      | 61.3%   |
| T02 Heart          | Done           | 93.0%     | 87.4% | 91.5%     | 94.5%   |
| T03 Liver          | Still training | 65.6%     | 59.3% | __        | 84.5%   |
| T04 Hippocampus    | Done           | 89.1%     | 87.0% | 88.6%     | 89.5%   |
| T05 Prostate       | Done           | 76.4%     | 67.1% | 72.0%     | 83.0%   |
| T06 Lung           | Done           | 67.6%     | 17.3% | 27.1%     | 69.0%   |
| T07 Pancreas       | Done           | 58.1%     | 24.2% | 63.6%     | 66.0%   |
| T08 Hepatic Vessel | Done           | 15.0%     | 38.6% | 58.7%     | 66.0%   |
| T09 Spleen         | Done           | 96.5%     | 52.6% | 77.5%     | 96.0%   |
| T10 Colon          | Done           | 34.0%     | 13.6% | 19.7%     | 56%     |


| Model | Data | Status| DSC | Ref DSC | Issue |
|-------|----|---|---------|---------|-----|
| Soft-L1-ACE + Dice + CE | T01 Brain Tumour | Poor | 71.3% | [84%](https://arxiv.org/abs/1709.00382) |
| Soft-L1-ACE + Dice + CE | T02 Heart | Fail | 17.2% | 94.5% |
| Soft-L1-ACE + Dice + CE | T03 Liver | Poor | 67.1% | 84.5% |  |
| Soft-L1-ACE + Dice + CE | T04 Hippocampus| Success | 89.2% | 90% |
| Soft-L1-ACE + Dice + CE | T05 Prostate |  Poor | 67.6% | 83% |
| Soft-L1-ACE + Dice + CE | T06 Lung | Fail | 0.0% | 69% |  |
| Soft-L1-ACE + Dice + CE | T07 Pancreas | Fail | 0.0% | 66% |
| Soft-L1-ACE + Dice + CE | T08 Hepatic Vessel | Success | 56.1% | 66% |
| Soft-L1-ACE + Dice + CE | T09 Spleen | Poor | 53.9% | 98% |  |
| Soft-L1-ACE + Dice + CE | T10 Colon | Failed | 0.0% | 62% | |