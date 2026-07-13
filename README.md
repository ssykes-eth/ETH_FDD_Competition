# ETH_FDD_Competition
https://www.kaggle.com/competitions/eth-fdd-competition/data

Steps
1. Download the data from Kaggle
2. Open the folder in VS Code
3. Use Claude
4. Share your project with GitHub

File Description
• X_train.csv, y_train.csv: the training set, including the features and labels
• X_test.csv: the testing set (make predictions based on this file)
• sample.csv: a sample submission file in the correct format

Task 1: Age Prediction
    We have modified the derived input data in three ways:
    • Irrelevant features
    • Outliers
    • Perturbations (e.g. missing values, etc)

    Subtask 0: Filling Missing Values
        • Background: There are missing values in the data
            • Originally they are set to NaN
            • Most methods cannot handle NaNs automatically
            • Different possible strategies to impute missing values: mean, median, most frequent etc.
        • Task requirements
            • We require that students fill missing values in the training and the test set

    Subtask 1: Outlier Detection 
        • Background: In the training set, there exists outliers
            • If the resulting model is not robust enough, it may be sensitive to the outliers
            • Solution: outlier removal
        • Task requirements
            • We require that students build an outlier detection model to make classification for samples in the training set i.e. whether they are outliers.

    Subtask 2: Feature Selection
        • Background: To make the task a bit more challenging, we added some manual features to the FreeSurfer-processed dataset.
            • Feature selection is needed
        • Advantages:
            • Simplifies the models to make them interpretable
            • Leads to shorter training times
            • Better generatlization by reducing overfitting
        • Task requirements
            • We require that students use feature selection methods to label the features as selected features and unselected features.
            • Here, unselected features includes irrelevant features and redundant features.

Main Task: Age Prediction
    • Background: After primary preprocessing and dimensionality reduction, now we finally arrive at the regression task.
    • Task requirements: We require that students use suitable regression methods to predict the age of a person from brain data.

Submission to Kaggle
• Kaggle link: https://www.kaggle.com/t/40276fd0e0c54a42bcc046758201dbd1 
• Team names must be alphanumeric (A-Z, a-z, 0-9).
• Deadline: 08/11/2026 11:59 PM
• Including public/private leaderboard:
    – Public leaderboard is available
    – Private leaderboard will open after the deadline
• Public baseline for passing the project: 0.5
• Project defense

Links
• Kaggle Competition: https://www.kaggle.com/competitions/eth-fdd-competition
• Slides: https://github.com/eth-fdd-fs26/FDD-WE1-public/blob/main/slides/Project%20week%201.pdf
• Lecture Notes: https://www.apollo-platform.xyz/notes/6f103c68-3b56-4657-8533-a01dd5e245ec
• Submission Script: https://github.com/eth-fdd-fs26/FDD-WE1-public/blob/main/submission.py