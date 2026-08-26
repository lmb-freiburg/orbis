# Binary Attention Probe - Per Source Class Performance Report

**Cached Features**: `./cached_features/cached_normalized_surprise_scores_3000.pt`  
**Total Samples**: `600`  

## 2. Summary Table Per Source Class (Accident Category)

| Class ID | Source Class Name | Total | Target (1/MC) | Normal/Other | Correct | Error | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1        | start_stop_or_stationary  | 11    | 6             | 5            | 8       | 3     | 72.73        | 80.00         | 66.67      | 72.73        | 0.7333 |
| 2        | moving_ahead_or_waiting   | 105   | 54            | 51           | 74      | 31    | 70.48        | 79.49         | 57.41      | 66.67        | 0.7778 |
| 3        | lateral                   | 80    | 36            | 44           | 58      | 22    | 72.50        | 69.44         | 69.44      | 69.44        | 0.7702 |
| 4        | oncoming                  | 52    | 27            | 25           | 38      | 14    | 73.08        | 80.95         | 62.96      | 70.83        | 0.8430 |
| 5        | turning                   | 243   | 117           | 126          | 171     | 72    | 70.37        | 73.68         | 59.83      | 66.04        | 0.7326 |
| 6        | pedestrian                | 14    | 8             | 6            | 9       | 5     | 64.29        | 71.43         | 62.50      | 66.67        | 0.6667 |
| 7        | obstacle                  | 10    | 4             | 6            | 8       | 2     | 80.00        | 66.67         | 100.00     | 80.00        | 0.8750 |
| 8        | leave_to_right            | 38    | 21            | 17           | 24      | 14    | 63.16        | 68.42         | 61.90      | 65.00        | 0.5966 |
| 9        | leave_to_left             | 47    | 29            | 18           | 30      | 17    | 63.83        | 75.00         | 62.07      | 67.92        | 0.7069 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ALL**   | **OVERALL VALIDATION**    | 600   | 302           | 298          | 420     | 180   | 70.00        | 74.21         | 61.92      | 67.51        | 0.7457 |

---

## 2. Combined Color-Coded Confusion Matrix Grid

![All Classes Confusion Matrix Grid](./confusion_matrices_surprise_combined_t3/cm_all_classes_grid.png)

---

## 3. Color-Coded Confusion Matrices Per Source Class

### Class 1: start_stop_or_stationary
**Total**: `11` | **Accuracy**: `72.73%` | **Correct**: `8` | **Error**: `3`  

![Confusion Matrix Class 1 - start_stop_or_stationary](./confusion_matrices_surprise_combined_t3/cm_class_1_start_stop_or_stationary.png)

---

### Class 2: moving_ahead_or_waiting
**Total**: `105` | **Accuracy**: `70.48%` | **Correct**: `74` | **Error**: `31`  

![Confusion Matrix Class 2 - moving_ahead_or_waiting](./confusion_matrices_surprise_combined_t3/cm_class_2_moving_ahead_or_waiting.png)

---

### Class 3: lateral
**Total**: `80` | **Accuracy**: `72.50%` | **Correct**: `58` | **Error**: `22`  

![Confusion Matrix Class 3 - lateral](./confusion_matrices_surprise_combined_t3/cm_class_3_lateral.png)

---

### Class 4: oncoming
**Total**: `52` | **Accuracy**: `73.08%` | **Correct**: `38` | **Error**: `14`  

![Confusion Matrix Class 4 - oncoming](./confusion_matrices_surprise_combined_t3/cm_class_4_oncoming.png)

---

### Class 5: turning
**Total**: `243` | **Accuracy**: `70.37%` | **Correct**: `171` | **Error**: `72`  

![Confusion Matrix Class 5 - turning](./confusion_matrices_surprise_combined_t3/cm_class_5_turning.png)

---

### Class 6: pedestrian
**Total**: `14` | **Accuracy**: `64.29%` | **Correct**: `9` | **Error**: `5`  

![Confusion Matrix Class 6 - pedestrian](./confusion_matrices_surprise_combined_t3/cm_class_6_pedestrian.png)

---

### Class 7: obstacle
**Total**: `10` | **Accuracy**: `80.00%` | **Correct**: `8` | **Error**: `2`  

![Confusion Matrix Class 7 - obstacle](./confusion_matrices_surprise_combined_t3/cm_class_7_obstacle.png)

---

### Class 8: leave_to_right
**Total**: `38` | **Accuracy**: `63.16%` | **Correct**: `24` | **Error**: `14`  

![Confusion Matrix Class 8 - leave_to_right](./confusion_matrices_surprise_combined_t3/cm_class_8_leave_to_right.png)

---

### Class 9: leave_to_left
**Total**: `47` | **Accuracy**: `63.83%` | **Correct**: `30` | **Error**: `17`  

![Confusion Matrix Class 9 - leave_to_left](./confusion_matrices_surprise_combined_t3/cm_class_9_leave_to_left.png)

---

## 4. Overall Color-Coded Confusion Matrix (Entire Variant)

**Total Validation Samples**: `600` | **Overall Accuracy**: `70.00%` | **Overall AUC**: `0.7457`  

![Overall Confusion Matrix](./confusion_matrices_surprise_combined_t3/cm_overall.png)

