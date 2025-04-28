# Predicting Stable Height of Block Stacks  
---
## Project Overview
This project was developed as part of the **Physical Reasoning Challenge** for the Computer Vision course at the University of Melbourne.
The objective is to predict the **stable height** of block stacks based on single images. The task involves identifying how many blocks in a stack are correctly placed without violating physical stability rules (center of mass support and flat surfaces).
The project was implemented using **Google Colab**.

---

## Approach
- **Baseline Model**:  
  - Built using **AlexNet** architecture.  
  - Implemented in the file: `Alexnet_classification.ipynb`
- **Final Model**:  
  - Improved model using **ResNet152**.  
  - Implemented in the file: `Resnet152_classification.ipynb`
- **Submission File**:  
  - The final prediction results are stored in `Final_Resnet152_batch8_prediction.csv`, formatted for Kaggle submission.

---

## Project Structure
```
├── data/                     # Folder containing training and test datasets (images)
    ├── 2025_Project_test.zip    # Test dataset file
    ├── 2025_Project_train.zip    # Training dataset file
├── util/                     # Folder containing utility code and supporting functions
    ├── Alexnet_classification.ipynb    # Baseline model using AlexNet
    ├── Resnet152_classification.ipynb  # Final model using ResNet152
    ├── Final_Resnet152_batch8_prediction.csv  # Final prediction results
    ├── README.md                 # Project documentation (this file)

---

## How to Run
1. Place the provided dataset inside the `data/` folder.
2. Run the appropriate `.ipynb` notebook (`Alexnet_classification.ipynb` or `Resnet152_classification.ipynb`) in **Google Colab**.
3. Adjust paths in the notebook if necessary to correctly locate `data/` and `util/` folders.
4. The final prediction will be saved in the format required for Kaggle submission.

---

## Notes
- Models were trained and evaluated on a custom subset of the **ShapeStacks** dataset provided for the assignment.
- No external pretrained models on the public ShapeStacks dataset were used, in compliance with project guidelines.
- Evaluation is based on **mean accuracy** over the test set.

---

## References
- Groth, O., Fuchs, F. B., Posner, I., & Vedaldi, A. (2018).  
  *ShapeStacks: Learning vision-based physical intuition for generalised object stacking.*  
  In Computer Vision – ECCV 2018 (pp. 724–739).

---
