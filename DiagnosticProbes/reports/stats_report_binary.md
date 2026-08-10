# Binary Attention Probe - Per Source Class Performance Report

**Cached Features**: `/Users/betterbrambola/Desktop/Desktop - Prafful’s MacBook Pro/UFRAssns/DLLabProject/orbis/DiagnosticProbes/cached_features/val_block18_all_correct_unpooled_mc.pt`  
**Total Samples**: `600`  

## 2. Summary Table Per Source Class (Accident Category)

| Class ID | Source Class Name | Total | Target (1/MC) | Normal/Other | Correct | Error | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1        | start_stop_or_stationary  | 11    | 6             | 5            | 11      | 0     | 100.00       | 100.00        | 100.00     | 100.00       | 1.0000 |
| 2        | moving_ahead_or_waiting   | 105   | 54            | 51           | 88      | 17    | 83.81        | 95.12         | 72.22      | 82.11        | 0.8972 |
| 3        | lateral                   | 80    | 36            | 44           | 67      | 13    | 83.75        | 96.00         | 66.67      | 78.69        | 0.8845 |
| 4        | oncoming                  | 52    | 27            | 25           | 41      | 11    | 78.85        | 83.33         | 74.07      | 78.43        | 0.8815 |
| 5        | turning                   | 243   | 117           | 126          | 180     | 63    | 74.07        | 78.12         | 64.10      | 70.42        | 0.8339 |
| 6        | pedestrian                | 14    | 8             | 6            | 10      | 4     | 71.43        | 83.33         | 62.50      | 71.43        | 0.7917 |
| 7        | obstacle                  | 10    | 4             | 6            | 9       | 1     | 90.00        | 100.00        | 75.00      | 85.71        | 1.0000 |
| 8        | leave_to_right            | 38    | 21            | 17           | 25      | 13    | 65.79        | 83.33         | 47.62      | 60.61        | 0.7983 |
| 9        | leave_to_left             | 47    | 29            | 18           | 30      | 17    | 63.83        | 92.86         | 44.83      | 60.47        | 0.8103 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ALL**   | **OVERALL VALIDATION**    | 600   | 302           | 298          | 461     | 139   | 76.83        | 85.90         | 64.57      | 73.72        | 0.8527 |

---

## 2. Combined Color-Coded Confusion Matrix Grid

![All Classes Confusion Matrix Grid](/Users/betterbrambola/Desktop/Desktop - Prafful’s MacBook Pro/UFRAssns/DLLabProject/orbis/DiagnosticProbes/confusionMatrices/binary/cm_all_classes_grid.png)

---

## 3. Color-Coded Confusion Matrices Per Source Class

### Class 1: start_stop_or_stationary
**Total**: `11` | **Accuracy**: `100.00%` | **Correct**: `11` | **Error**: `0`  

![Confusion Matrix Class 1 - start_stop_or_stationary](/Users/betterbrambola/Desktop/Desktop - Prafful’s MacBook Pro/UFRAssns/DLLabProject/orbis/DiagnosticProbes/confusionMatrices/binary/cm_class_1_start_stop_or_stationary.png)

---

### Class 2: moving_ahead_or_waiting
**Total**: `105` | **Accuracy**: `83.81%` | **Correct**: `88` | **Error**: `17`  

![Confusion Matrix Class 2 - moving_ahead_or_waiting](/Users/betterbrambola/Desktop/Desktop - Prafful’s MacBook Pro/UFRAssns/DLLabProject/orbis/DiagnosticProbes/confusionMatrices/binary/cm_class_2_moving_ahead_or_waiting.png)

---

### Class 3: lateral
**Total**: `80` | **Accuracy**: `83.75%` | **Correct**: `67` | **Error**: `13`  

![Confusion Matrix Class 3 - lateral](/Users/betterbrambola/Desktop/Desktop - Prafful’s MacBook Pro/UFRAssns/DLLabProject/orbis/DiagnosticProbes/confusionMatrices/binary/cm_class_3_lateral.png)

---

### Class 4: oncoming
**Total**: `52` | **Accuracy**: `78.85%` | **Correct**: `41` | **Error**: `11`  

![Confusion Matrix Class 4 - oncoming](/Users/betterbrambola/Desktop/Desktop - Prafful’s MacBook Pro/UFRAssns/DLLabProject/orbis/DiagnosticProbes/confusionMatrices/binary/cm_class_4_oncoming.png)

---

### Class 5: turning
**Total**: `243` | **Accuracy**: `74.07%` | **Correct**: `180` | **Error**: `63`  

![Confusion Matrix Class 5 - turning](/Users/betterbrambola/Desktop/Desktop - Prafful’s MacBook Pro/UFRAssns/DLLabProject/orbis/DiagnosticProbes/confusionMatrices/binary/cm_class_5_turning.png)

---

### Class 6: pedestrian
**Total**: `14` | **Accuracy**: `71.43%` | **Correct**: `10` | **Error**: `4`  

![Confusion Matrix Class 6 - pedestrian](/Users/betterbrambola/Desktop/Desktop - Prafful’s MacBook Pro/UFRAssns/DLLabProject/orbis/DiagnosticProbes/confusionMatrices/binary/cm_class_6_pedestrian.png)

---

### Class 7: obstacle
**Total**: `10` | **Accuracy**: `90.00%` | **Correct**: `9` | **Error**: `1`  

![Confusion Matrix Class 7 - obstacle](/Users/betterbrambola/Desktop/Desktop - Prafful’s MacBook Pro/UFRAssns/DLLabProject/orbis/DiagnosticProbes/confusionMatrices/binary/cm_class_7_obstacle.png)

---

### Class 8: leave_to_right
**Total**: `38` | **Accuracy**: `65.79%` | **Correct**: `25` | **Error**: `13`  

![Confusion Matrix Class 8 - leave_to_right](/Users/betterbrambola/Desktop/Desktop - Prafful’s MacBook Pro/UFRAssns/DLLabProject/orbis/DiagnosticProbes/confusionMatrices/binary/cm_class_8_leave_to_right.png)

---

### Class 9: leave_to_left
**Total**: `47` | **Accuracy**: `63.83%` | **Correct**: `30` | **Error**: `17`  

![Confusion Matrix Class 9 - leave_to_left](/Users/betterbrambola/Desktop/Desktop - Prafful’s MacBook Pro/UFRAssns/DLLabProject/orbis/DiagnosticProbes/confusionMatrices/binary/cm_class_9_leave_to_left.png)

---

## 4. Overall Color-Coded Confusion Matrix (Entire Variant)

**Total Validation Samples**: `600` | **Overall Accuracy**: `76.83%` | **Overall AUC**: `0.8527`  

![Overall Confusion Matrix](/Users/betterbrambola/Desktop/Desktop - Prafful’s MacBook Pro/UFRAssns/DLLabProject/orbis/DiagnosticProbes/confusionMatrices/binary/cm_overall.png)

