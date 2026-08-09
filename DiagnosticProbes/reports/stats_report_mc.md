# Multiclass (MC) Attention Probe - Per Source Class Performance Report

**Cached Features**: `./cached_features/val_block18_all_correct_unpooled_mc.pt`  
**Total Samples**: `600`  

## 1. 10x10 Multiclass Confusion Matrix Heatmap

![10x10 Multiclass Confusion Matrix Heatmap](./confusion_matrices_mc/cm_10x10_multiclass_heatmap.png)

---

## 2. Summary Table Per Source Class (Accident Category)

| Class ID | Source Class Name | Total | Target (1/MC) | Normal/Other | Correct | Error | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1        | start_stop_or_stationary  | 11    | 6             | 5            | 5       | 6     | 45.45        | 12.50         | 20.00      | 15.38        | N/A   |
| 2        | moving_ahead_or_waiting   | 105   | 54            | 51           | 75      | 30    | 71.43        | 28.68         | 24.04      | 24.56        | N/A   |
| 3        | lateral                   | 80    | 36            | 44           | 50      | 30    | 62.50        | 27.31         | 19.61      | 19.93        | N/A   |
| 4        | oncoming                  | 52    | 27            | 25           | 28      | 24    | 53.85        | 24.55         | 15.79      | 15.99        | N/A   |
| 5        | turning                   | 243   | 117           | 126          | 163     | 80    | 67.08        | 18.20         | 16.60      | 16.92        | N/A   |
| 6        | pedestrian                | 14    | 8             | 6            | 7       | 7     | 50.00        | 58.33         | 37.50      | 35.98        | N/A   |
| 7        | obstacle                  | 10    | 4             | 6            | 6       | 4     | 60.00        | 57.14         | 36.11      | 38.97        | N/A   |
| 8        | leave_to_right            | 38    | 21            | 17           | 17      | 21    | 44.74        | 24.75         | 16.48      | 12.18        | N/A   |
| 9        | leave_to_left             | 47    | 29            | 18           | 17      | 30    | 36.17        | 6.75          | 13.49      | 8.99         | N/A   |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ALL**   | **OVERALL VALIDATION**    | 600   | 302           | 298          | 368     | 232   | 61.33        | 37.00         | 26.96      | 29.56        | 0.8007 |

---

## 2. Combined Color-Coded Confusion Matrix Grid

![All Classes Confusion Matrix Grid](./confusion_matrices_mc/cm_all_classes_grid_mc.png)

---

## 3. Color-Coded Confusion Matrices Per Source Class

### Class 1: start_stop_or_stationary
**Total**: `11` | **Accuracy**: `45.45%` | **Correct**: `5` | **Error**: `6`  

![Confusion Matrix Class 1 - start_stop_or_stationary](./confusion_matrices_mc/cm_class_1_start_stop_or_stationary_mc.png)

---

### Class 2: moving_ahead_or_waiting
**Total**: `105` | **Accuracy**: `71.43%` | **Correct**: `75` | **Error**: `30`  

![Confusion Matrix Class 2 - moving_ahead_or_waiting](./confusion_matrices_mc/cm_class_2_moving_ahead_or_waiting_mc.png)

---

### Class 3: lateral
**Total**: `80` | **Accuracy**: `62.50%` | **Correct**: `50` | **Error**: `30`  

![Confusion Matrix Class 3 - lateral](./confusion_matrices_mc/cm_class_3_lateral_mc.png)

---

### Class 4: oncoming
**Total**: `52` | **Accuracy**: `53.85%` | **Correct**: `28` | **Error**: `24`  

![Confusion Matrix Class 4 - oncoming](./confusion_matrices_mc/cm_class_4_oncoming_mc.png)

---

### Class 5: turning
**Total**: `243` | **Accuracy**: `67.08%` | **Correct**: `163` | **Error**: `80`  

![Confusion Matrix Class 5 - turning](./confusion_matrices_mc/cm_class_5_turning_mc.png)

---

### Class 6: pedestrian
**Total**: `14` | **Accuracy**: `50.00%` | **Correct**: `7` | **Error**: `7`  

![Confusion Matrix Class 6 - pedestrian](./confusion_matrices_mc/cm_class_6_pedestrian_mc.png)

---

### Class 7: obstacle
**Total**: `10` | **Accuracy**: `60.00%` | **Correct**: `6` | **Error**: `4`  

![Confusion Matrix Class 7 - obstacle](./confusion_matrices_mc/cm_class_7_obstacle_mc.png)

---

### Class 8: leave_to_right
**Total**: `38` | **Accuracy**: `44.74%` | **Correct**: `17` | **Error**: `21`  

![Confusion Matrix Class 8 - leave_to_right](./confusion_matrices_mc/cm_class_8_leave_to_right_mc.png)

---

### Class 9: leave_to_left
**Total**: `47` | **Accuracy**: `36.17%` | **Correct**: `17` | **Error**: `30`  

![Confusion Matrix Class 9 - leave_to_left](./confusion_matrices_mc/cm_class_9_leave_to_left_mc.png)

---

## 4. Overall Color-Coded Confusion Matrix (Entire Variant)

**Total Validation Samples**: `600` | **Overall Accuracy**: `61.33%` | **Overall AUC**: `0.8007`  

![Overall Confusion Matrix](./confusion_matrices_mc/cm_overall_mc.png)

