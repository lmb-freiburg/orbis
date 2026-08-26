# Binary Attention Probe - Per Source Class Performance Report

**Cached Features**: `./cached_features/val_block18_3600_correct_unpooled_mc.pt`  
**Total Samples**: `720`  

## 1. Summary Table Per Source Class (Accident Category)

| Class ID | Source Class Name | Total | Normal (0) | Anom (1) | TN | FP | FN | TP | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1        | start_stop_or_stationary  | 13    | 5          | 8        | 4  | 1  | 2  | 6  | 76.92        | 85.71         | 75.00      | 80.00        | 0.8250 |
| 2        | moving_ahead_or_waiting   | 116   | 61         | 55       | 44 | 17 | 10 | 45 | 76.72        | 72.58         | 81.82      | 76.92        | 0.8811 |
| 3        | lateral                   | 111   | 49         | 62       | 35 | 14 | 16 | 46 | 72.97        | 76.67         | 74.19      | 75.41        | 0.7962 |
| 4        | oncoming                  | 72    | 31         | 41       | 25 | 6  | 11 | 30 | 76.39        | 83.33         | 73.17      | 77.92        | 0.8749 |
| 5        | turning                   | 262   | 111        | 151      | 84 | 27 | 32 | 119 | 77.48        | 81.51         | 78.81      | 80.13        | 0.8463 |
| 6        | pedestrian                | 13    | 9          | 4        | 6  | 3  | 3  | 1  | 53.85        | 25.00         | 25.00      | 25.00        | 0.4722 |
| 7        | obstacle                  | 18    | 9          | 9        | 8  | 1  | 5  | 4  | 66.67        | 80.00         | 44.44      | 57.14        | 0.7160 |
| 8        | leave_to_right            | 56    | 30         | 26       | 25 | 5  | 8  | 18 | 76.79        | 78.26         | 69.23      | 73.47        | 0.8256 |
| 9        | leave_to_left             | 59    | 18         | 41       | 15 | 3  | 21 | 20 | 59.32        | 86.96         | 48.78      | 62.50        | 0.7575 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ALL**   | **OVERALL VALIDATION**    | 720   | 323        | 397      | 246 | 77 | 108 | 289 | 74.31        | 78.96         | 72.80      | 75.75        | 0.8233 |

---

## 2. Combined Color-Coded Confusion Matrix Grid

![All Classes Confusion Matrix Grid](./confusion_matrices/cm_all_classes_grid.png)

---

## 3. Color-Coded Confusion Matrices Per Source Class

### Class 1: start_stop_or_stationary
**Total**: `13` | **Normal**: `5` | **Anomalous**: `8` | **Accuracy**: `76.92%` | **TN**: `4` | **FP**: `1` | **FN**: `2` | **TP**: `6`  

![Confusion Matrix Class 1 - start_stop_or_stationary](./confusion_matrices/cm_class_1_start_stop_or_stationary.png)

---

### Class 2: moving_ahead_or_waiting
**Total**: `116` | **Normal**: `61` | **Anomalous**: `55` | **Accuracy**: `76.72%` | **TN**: `44` | **FP**: `17` | **FN**: `10` | **TP**: `45`  

![Confusion Matrix Class 2 - moving_ahead_or_waiting](./confusion_matrices/cm_class_2_moving_ahead_or_waiting.png)

---

### Class 3: lateral
**Total**: `111` | **Normal**: `49` | **Anomalous**: `62` | **Accuracy**: `72.97%` | **TN**: `35` | **FP**: `14` | **FN**: `16` | **TP**: `46`  

![Confusion Matrix Class 3 - lateral](./confusion_matrices/cm_class_3_lateral.png)

---

### Class 4: oncoming
**Total**: `72` | **Normal**: `31` | **Anomalous**: `41` | **Accuracy**: `76.39%` | **TN**: `25` | **FP**: `6` | **FN**: `11` | **TP**: `30`  

![Confusion Matrix Class 4 - oncoming](./confusion_matrices/cm_class_4_oncoming.png)

---

### Class 5: turning
**Total**: `262` | **Normal**: `111` | **Anomalous**: `151` | **Accuracy**: `77.48%` | **TN**: `84` | **FP**: `27` | **FN**: `32` | **TP**: `119`  

![Confusion Matrix Class 5 - turning](./confusion_matrices/cm_class_5_turning.png)

---

### Class 6: pedestrian
**Total**: `13` | **Normal**: `9` | **Anomalous**: `4` | **Accuracy**: `53.85%` | **TN**: `6` | **FP**: `3` | **FN**: `3` | **TP**: `1`  

![Confusion Matrix Class 6 - pedestrian](./confusion_matrices/cm_class_6_pedestrian.png)

---

### Class 7: obstacle
**Total**: `18` | **Normal**: `9` | **Anomalous**: `9` | **Accuracy**: `66.67%` | **TN**: `8` | **FP**: `1` | **FN**: `5` | **TP**: `4`  

![Confusion Matrix Class 7 - obstacle](./confusion_matrices/cm_class_7_obstacle.png)

---

### Class 8: leave_to_right
**Total**: `56` | **Normal**: `30` | **Anomalous**: `26` | **Accuracy**: `76.79%` | **TN**: `25` | **FP**: `5` | **FN**: `8` | **TP**: `18`  

![Confusion Matrix Class 8 - leave_to_right](./confusion_matrices/cm_class_8_leave_to_right.png)

---

### Class 9: leave_to_left
**Total**: `59` | **Normal**: `18` | **Anomalous**: `41` | **Accuracy**: `59.32%` | **TN**: `15` | **FP**: `3` | **FN**: `21` | **TP**: `20`  

![Confusion Matrix Class 9 - leave_to_left](./confusion_matrices/cm_class_9_leave_to_left.png)

---

## 4. Overall Color-Coded Confusion Matrix (Entire Variant)

**Total Validation Samples**: `720` | **Overall Accuracy**: `74.31%` | **Overall AUC**: `0.8233` | **TN**: `246` | **FP**: `77` | **FN**: `108` | **TP**: `289`  

![Overall Confusion Matrix](./confusion_matrices/cm_overall.png)

