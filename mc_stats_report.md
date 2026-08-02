# Multiclass (MC) Attention Probe - Per Source Class Performance Report

**Cached Features**: `./cached_features/val_block18_3600_correct_unpooled_mc.pt`  
**Total Samples**: `720`  

## 1. Summary Table Per Source Class (Accident Category)

| Class ID | Source Class Name | Total | Target (1/MC) | Normal/Other | Correct | Error | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1        | start_stop_or_stationary  | 13    | 3             | 10           | 3       | 10    | 23.08        | 14.29         | 6.43       | 8.84         | N/A   |
| 2        | moving_ahead_or_waiting   | 116   | 37            | 79           | 37      | 79    | 31.90        | 17.69         | 7.22       | 10.05        | N/A   |
| 3        | lateral                   | 111   | 37            | 74           | 37      | 74    | 33.33        | 12.17         | 7.20       | 8.99         | N/A   |
| 4        | oncoming                  | 72    | 23            | 49           | 23      | 49    | 31.94        | 13.86         | 6.93       | 9.24         | N/A   |
| 5        | turning                   | 262   | 84            | 178          | 84      | 178   | 32.06        | 13.88         | 6.59       | 8.88         | N/A   |
| 6        | pedestrian                | 13    | 4             | 9            | 4       | 9     | 30.77        | 13.33         | 7.41       | 9.52         | N/A   |
| 7        | obstacle                  | 18    | 8             | 10           | 8       | 10    | 44.44        | 22.92         | 11.11      | 14.58        | N/A   |
| 8        | leave_to_right            | 56    | 9             | 47           | 9       | 47    | 16.07        | 9.94          | 3.94       | 5.63         | N/A   |
| 9        | leave_to_left             | 59    | 19            | 40           | 19      | 40    | 32.20        | 13.26         | 7.23       | 9.15         | N/A   |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ALL**   | **OVERALL VALIDATION**    | 720   | 224           | 496          | 224     | 496   | 31.11        | 46.19         | 31.11      | 34.80        | 0.7558 |

---

## 2. Combined Color-Coded Confusion Matrix Grid

![All Classes Confusion Matrix Grid](./confusion_matrices_mc/cm_all_classes_grid_mc.png)

---

## 3. Color-Coded Confusion Matrices Per Source Class

### Class 1: start_stop_or_stationary
**Total**: `13` | **Accuracy**: `23.08%` | **Correct**: `3` | **Error**: `10`  

![Confusion Matrix Class 1 - start_stop_or_stationary](./confusion_matrices_mc/cm_class_1_start_stop_or_stationary_mc.png)

---

### Class 2: moving_ahead_or_waiting
**Total**: `116` | **Accuracy**: `31.90%` | **Correct**: `37` | **Error**: `79`  

![Confusion Matrix Class 2 - moving_ahead_or_waiting](./confusion_matrices_mc/cm_class_2_moving_ahead_or_waiting_mc.png)

---

### Class 3: lateral
**Total**: `111` | **Accuracy**: `33.33%` | **Correct**: `37` | **Error**: `74`  

![Confusion Matrix Class 3 - lateral](./confusion_matrices_mc/cm_class_3_lateral_mc.png)

---

### Class 4: oncoming
**Total**: `72` | **Accuracy**: `31.94%` | **Correct**: `23` | **Error**: `49`  

![Confusion Matrix Class 4 - oncoming](./confusion_matrices_mc/cm_class_4_oncoming_mc.png)

---

### Class 5: turning
**Total**: `262` | **Accuracy**: `32.06%` | **Correct**: `84` | **Error**: `178`  

![Confusion Matrix Class 5 - turning](./confusion_matrices_mc/cm_class_5_turning_mc.png)

---

### Class 6: pedestrian
**Total**: `13` | **Accuracy**: `30.77%` | **Correct**: `4` | **Error**: `9`  

![Confusion Matrix Class 6 - pedestrian](./confusion_matrices_mc/cm_class_6_pedestrian_mc.png)

---

### Class 7: obstacle
**Total**: `18` | **Accuracy**: `44.44%` | **Correct**: `8` | **Error**: `10`  

![Confusion Matrix Class 7 - obstacle](./confusion_matrices_mc/cm_class_7_obstacle_mc.png)

---

### Class 8: leave_to_right
**Total**: `56` | **Accuracy**: `16.07%` | **Correct**: `9` | **Error**: `47`  

![Confusion Matrix Class 8 - leave_to_right](./confusion_matrices_mc/cm_class_8_leave_to_right_mc.png)

---

### Class 9: leave_to_left
**Total**: `59` | **Accuracy**: `32.20%` | **Correct**: `19` | **Error**: `40`  

![Confusion Matrix Class 9 - leave_to_left](./confusion_matrices_mc/cm_class_9_leave_to_left_mc.png)

---

## 4. Overall Color-Coded Confusion Matrix (Entire Variant)

**Total Validation Samples**: `720` | **Overall Accuracy**: `31.11%` | **Overall AUC**: `0.7558`  

![Overall Confusion Matrix](./confusion_matrices_mc/cm_overall_mc.png)

