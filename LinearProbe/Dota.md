-> Number of entire video sequences - 4,677 videos with temporal, spatial, and categorical annotations \
-> Sampling frequency - Original 30 fps, but extracted frames are at 10 fps in the dataset. Resampling data @ 5 fps to stay consistent with ORBIS \
-> Resolution of frames - 1280 x 720 \
-> Sample gifs which are being used for analysis can be found [here](https://drive.google.com/drive/folders/1hQK9_jdduxfz3JqFivd7eLjolUWyxjWK?usp=share_link)


-> Data Preprocessing - Captured non-OOD data from the first frames (5 sub-sampled) of each sequence, and the anamolous data from the first few frames (5 subsampled) of OOD data. Ignored if the length of anamoly is shorter than the required number of frames. For non-OOD samples, we sampled from the last non-anamolous frames of the sequence if the beginning of the video frames doesn't have enough number of required frames. \
-> Number of Valid datapoints collected after data preprocessing - 4572 (non-OOD) + 4577 (OOD) \
-> 80:20 split for train:val 
![alt text](dota.png)


--- Multiclass Distribution (Capped at 4000 samples) ---
 0                    | ID: 0   | Count: 2008 - ID
 1                    | ID: 1   | Count: 42
 2                    | ID: 2   | Count: 316
 3                    | ID: 3   | Count: 356
 4                    | ID: 4   | Count: 199
 5                    | ID: 5   | Count: 806
 6                    | ID: 6   | Count: 52
 7                    | ID: 7   | Count: 46
 8                    | ID: 8   | Count: 175
-------------------------------
Attention heatmaps:
1. Min-max normalization to 0-1


Multi Class 
1. AUC Measure - ovr - One vs Rest
2. Accruacy,Precision, Recall - 