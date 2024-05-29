# Structure
----
```
freq_based_deepfake
 ┣ datasets
 ┃ ┣ test
 ┃ ┃ ┣ 001.jpg
 ┃ ┃ ┣ 500.jpg
 ┃ ┃ ┗ test_labels.csv
 ┃ ┗ train
 ┃ ┃ ┣ generated
 ┃ ┃ ┃ ┣ samples_0.png
 ┃ ┃ ┃ ┗ samples_999.png
 ┃ ┃ ┗ real
 ┃ ┃ ┃ ┣ 00000.jpg
 ┃ ┃ ┃ ┗ 69999.jpg
 ┣ src
 ┃ ┣ Configueration.py
 ┃ ┣ CustomDataLoader.py
 ┃ ┣ FreqEncoder.py
 ┃ ┗ HighFreqVit.py
 ┣ .gitignore
 ┣ evaluation.py
 ┣ model_state_dict.pt
 ┣ README.md
 ┗ train.py
```

# Train
----
`python train.py'

# Evaluation
----
`python evaluation.py'
