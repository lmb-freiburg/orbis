# Binary Attention Probe - Per Source Class Performance Report

**Cached Features**: `./cached_features/val_block18_all_correct_unpooled_mc.pt`  
**Total Samples**: `720`  

## 1. Summary Table Per Source Class (Accident Category)

| Class ID | Source Class Name | Total | Normal (0) | Anom (1) | TN | FP | FN | TP | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1        | start_stop_or_stationary  | 13    | 5          | 8        | 4  | 1  | 1  | 7  | 84.62        | 87.50         | 87.50      | 87.50        | 0.8750 |
| 2        | moving_ahead_or_waiting   | 116   | 61         | 55       | 42 | 19 | 10 | 45 | 75.00        | 70.31         | 81.82      | 75.63        | 0.8548 |
| 3        | lateral                   | 111   | 49         | 62       | 30 | 19 | 15 | 47 | 69.37        | 71.21         | 75.81      | 73.44        | 0.7883 |
| 4        | oncoming                  | 72    | 31         | 41       | 25 | 6  | 6  | 35 | 83.33        | 85.37         | 85.37      | 85.37        | 0.8647 |
| 5        | turning                   | 262   | 111        | 151      | 80 | 31 | 24 | 127 | 79.01        | 80.38         | 84.11      | 82.20        | 0.8585 |
| 6        | pedestrian                | 13    | 9          | 4        | 7  | 2  | 3  | 1  | 61.54        | 33.33         | 25.00      | 28.57        | 0.5556 |
| 7        | obstacle                  | 18    | 9          | 9        | 5  | 4  | 3  | 6  | 61.11        | 60.00         | 66.67      | 63.16        | 0.6543 |
| 8        | leave_to_right            | 56    | 30         | 26       | 24 | 6  | 6  | 20 | 78.57        | 76.92         | 76.92      | 76.92        | 0.8462 |
| 9        | leave_to_left             | 59    | 18         | 41       | 16 | 2  | 17 | 24 | 67.80        | 92.31         | 58.54      | 71.64        | 0.7886 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ALL**   | **OVERALL VALIDATION**    | 720   | 323        | 397      | 233 | 90 | 85 | 312 | 75.69        | 77.61         | 78.59      | 78.10        | 0.8266 |

---

## 2. Combined Color-Coded Confusion Matrix Grid

![All Classes Confusion Matrix Grid](./confusion_matrices/cm_all_classes_grid.png)

---

## 3. Color-Coded Confusion Matrices Per Source Class

### Class 1: start_stop_or_stationary
**Total**: `13` | **Normal**: `5` | **Anomalous**: `8` | **Accuracy**: `84.62%` | **TN**: `4` | **FP**: `1` | **FN**: `1` | **TP**: `7`  

![Confusion Matrix Class 1 - start_stop_or_stationary](./confusion_matrices/cm_class_1_start_stop_or_stationary.png)

---

### Class 2: moving_ahead_or_waiting
**Total**: `116` | **Normal**: `61` | **Anomalous**: `55` | **Accuracy**: `75.00%` | **TN**: `42` | **FP**: `19` | **FN**: `10` | **TP**: `45`  

![Confusion Matrix Class 2 - moving_ahead_or_waiting](./confusion_matrices/cm_class_2_moving_ahead_or_waiting.png)

---

### Class 3: lateral
**Total**: `111` | **Normal**: `49` | **Anomalous**: `62` | **Accuracy**: `69.37%` | **TN**: `30` | **FP**: `19` | **FN**: `15` | **TP**: `47`  

![Confusion Matrix Class 3 - lateral](./confusion_matrices/cm_class_3_lateral.png)

---

### Class 4: oncoming
**Total**: `72` | **Normal**: `31` | **Anomalous**: `41` | **Accuracy**: `83.33%` | **TN**: `25` | **FP**: `6` | **FN**: `6` | **TP**: `35`  

![Confusion Matrix Class 4 - oncoming](./confusion_matrices/cm_class_4_oncoming.png)

---

### Class 5: turning
**Total**: `262` | **Normal**: `111` | **Anomalous**: `151` | **Accuracy**: `79.01%` | **TN**: `80` | **FP**: `31` | **FN**: `24` | **TP**: `127`  

![Confusion Matrix Class 5 - turning](./confusion_matrices/cm_class_5_turning.png)

---

### Class 6: pedestrian
**Total**: `13` | **Normal**: `9` | **Anomalous**: `4` | **Accuracy**: `61.54%` | **TN**: `7` | **FP**: `2` | **FN**: `3` | **TP**: `1`  

![Confusion Matrix Class 6 - pedestrian](./confusion_matrices/cm_class_6_pedestrian.png)

---

### Class 7: obstacle
**Total**: `18` | **Normal**: `9` | **Anomalous**: `9` | **Accuracy**: `61.11%` | **TN**: `5` | **FP**: `4` | **FN**: `3` | **TP**: `6`  

![Confusion Matrix Class 7 - obstacle](./confusion_matrices/cm_class_7_obstacle.png)

---

### Class 8: leave_to_right
**Total**: `56` | **Normal**: `30` | **Anomalous**: `26` | **Accuracy**: `78.57%` | **TN**: `24` | **FP**: `6` | **FN**: `6` | **TP**: `20`  

![Confusion Matrix Class 8 - leave_to_right](./confusion_matrices/cm_class_8_leave_to_right.png)

---

### Class 9: leave_to_left
**Total**: `59` | **Normal**: `18` | **Anomalous**: `41` | **Accuracy**: `67.80%` | **TN**: `16` | **FP**: `2` | **FN**: `17` | **TP**: `24`  

![Confusion Matrix Class 9 - leave_to_left](./confusion_matrices/cm_class_9_leave_to_left.png)

---

## 4. Overall Color-Coded Confusion Matrix (Entire Variant)

**Total Validation Samples**: `720` | **Overall Accuracy**: `75.69%` | **Overall AUC**: `0.8266` | **TN**: `233` | **FP**: `90` | **FN**: `85` | **TP**: `312`  

![Overall Confusion Matrix](./confusion_matrices/cm_overall.png)

