# Multiclass (MC) Attention Probe - Per Source Class Performance Report

**Cached Features**: `./cached_features/val_block18_3600_correct_unpooled_mc.pt`  
**Total Samples**: `720`  

## 1. Summary Table Per Source Class (Accident Category)

| Class ID | Source Class Name | Total | Target (1/MC) | Normal/Other | Correct | Error | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1        | start_stop_or_stationary  | 13    | 4             | 9            | 4       | 9     | 30.77        | 11.11         | 20.00      | 14.29        | N/A   |
| 2        | moving_ahead_or_waiting   | 116   | 77            | 39           | 77      | 39    | 66.38        | 34.13         | 26.10      | 28.06        | N/A   |
| 3        | lateral                   | 111   | 58            | 53           | 58      | 53    | 52.25        | 26.62         | 18.59      | 19.14        | N/A   |
| 4        | oncoming                  | 72    | 34            | 38           | 34      | 38    | 47.22        | 28.38         | 17.23      | 18.19        | N/A   |
| 5        | turning                   | 262   | 176           | 86           | 176     | 86    | 67.18        | 25.38         | 23.24      | 23.38        | N/A   |
| 6        | pedestrian                | 13    | 8             | 5            | 8       | 5     | 61.54        | 18.18         | 22.22      | 20.00        | N/A   |
| 7        | obstacle                  | 18    | 8             | 10           | 8       | 10    | 44.44        | 13.33         | 22.22      | 16.67        | N/A   |
| 8        | leave_to_right            | 56    | 27            | 29           | 27      | 29    | 48.21        | 32.68         | 18.10      | 16.13        | N/A   |
| 9        | leave_to_left             | 59    | 21            | 38           | 21      | 38    | 35.59        | 19.63         | 17.37      | 11.89        | N/A   |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ALL**   | **OVERALL VALIDATION**    | 720   | 413           | 307          | 413     | 307   | 57.36        | 52.88         | 57.36      | 52.97        | 0.7632 |

---

## 2. Combined Color-Coded Confusion Matrix Grid

![All Classes Confusion Matrix Grid](./confusion_matrices_mc/cm_all_classes_grid_mc.png)

---

## 3. Color-Coded Confusion Matrices Per Source Class

### Class 1: start_stop_or_stationary
**Total**: `13` | **Accuracy**: `30.77%` | **Correct**: `4` | **Error**: `9`  

![Confusion Matrix Class 1 - start_stop_or_stationary](./confusion_matrices_mc/cm_class_1_start_stop_or_stationary_mc.png)

---

### Class 2: moving_ahead_or_waiting
**Total**: `116` | **Accuracy**: `66.38%` | **Correct**: `77` | **Error**: `39`  

![Confusion Matrix Class 2 - moving_ahead_or_waiting](./confusion_matrices_mc/cm_class_2_moving_ahead_or_waiting_mc.png)

---

### Class 3: lateral
**Total**: `111` | **Accuracy**: `52.25%` | **Correct**: `58` | **Error**: `53`  

![Confusion Matrix Class 3 - lateral](./confusion_matrices_mc/cm_class_3_lateral_mc.png)

---

### Class 4: oncoming
**Total**: `72` | **Accuracy**: `47.22%` | **Correct**: `34` | **Error**: `38`  

![Confusion Matrix Class 4 - oncoming](./confusion_matrices_mc/cm_class_4_oncoming_mc.png)

---

### Class 5: turning
**Total**: `262` | **Accuracy**: `67.18%` | **Correct**: `176` | **Error**: `86`  

![Confusion Matrix Class 5 - turning](./confusion_matrices_mc/cm_class_5_turning_mc.png)

---

### Class 6: pedestrian
**Total**: `13` | **Accuracy**: `61.54%` | **Correct**: `8` | **Error**: `5`  

![Confusion Matrix Class 6 - pedestrian](./confusion_matrices_mc/cm_class_6_pedestrian_mc.png)

---

### Class 7: obstacle
**Total**: `18` | **Accuracy**: `44.44%` | **Correct**: `8` | **Error**: `10`  

![Confusion Matrix Class 7 - obstacle](./confusion_matrices_mc/cm_class_7_obstacle_mc.png)

---

### Class 8: leave_to_right
**Total**: `56` | **Accuracy**: `48.21%` | **Correct**: `27` | **Error**: `29`  

![Confusion Matrix Class 8 - leave_to_right](./confusion_matrices_mc/cm_class_8_leave_to_right_mc.png)

---

### Class 9: leave_to_left
**Total**: `59` | **Accuracy**: `35.59%` | **Correct**: `21` | **Error**: `38`  

![Confusion Matrix Class 9 - leave_to_left](./confusion_matrices_mc/cm_class_9_leave_to_left_mc.png)

---

## 4. Overall Color-Coded Confusion Matrix (Entire Variant)

**Total Validation Samples**: `720` | **Overall Accuracy**: `57.36%` | **Overall AUC**: `0.7632`  

![Overall Confusion Matrix](./confusion_matrices_mc/cm_overall_mc.png)

