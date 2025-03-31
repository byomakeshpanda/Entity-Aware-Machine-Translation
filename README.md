# Entity-Aware Machine Translation (EA-MT)
## Table of Contents
- [Overview](#Overview)  
- [Steps for Inference](#steps-for-inference)  
- [Results](#results)  
- [References](#references)  
## Overview
This project implements Entity-Aware Machine Translation (EA-MT) using Flan-T5 with multi-task learning for **English** to **French** language. The model is trained simultaneously on Named Entity Recognition (NER) and EA-MT tasks, with a higher weight given to the EA-MT task in the loss function. This approach improves translation quality, particularly for rare words and named entities.


## Steps for Inference

1. **Clone the Repository**  
   Clone the GitHub repository and navigate to the project directory or you can download the entire repo in zip format:  
   ```bash
   git clone https://github.com/byomakeshpanda/Entity-Aware-Machine-Translation.git

   cd Entity-Aware-Machine-Translation
2. **Create and Activate a Conda Environment**
    Create a new Conda environment and activate it:
    ``` bash
    conda create --name ea-mt python=3.12.9 -y
    conda activate ea-mt
3. **Install Dependencies**

    Install all required dependencies:
    ``` bash
    pip install -r requirements.txt
4. **Download the Pretrained Model**  

- Download the model from [link](https://drive.google.com/drive/folders/1XdpwfXuycTRrWtNLEJjf9PVsCLrEvgAL?usp=drive_link).  
- Extract the ZIP file and rename the extracted folder to `t5_large_finetuned`.
- Move the `t5_large_finetuned` folder to `src\model` path. 
- Ensure you are inside the `Entity-Aware-Machine-Translation` directory before proceeding.

5. **Run Inference** 

To randomly sample given number of test samples and compute the BLEU score run the following command:  

     python src/inference/predict_test.py

To input an English sentence and retrieve entity aware translated french sentence run the following command:


    python src/inference/predict_sample.py

### Results  
Achieved BLEU score of **44.7**.

## References

 - [Entity-aware Multi-task Training Helps Rare Word Machine Translation](https://aclanthology.org/2024.inlg-main.5.pdf)
 - [ Extract and Attend: Improving Entity Translation in Neural Machine Translation.](https://arxiv.org/abs/2306.02242)
 - [SemEval 2025 EA-MT Dataset and Task.](https://sapienzanlp.github.io/ea-mt/)
 - [End-to-End Entity-Aware Neural Machine Translation.](https://www.researchgate.net/publication/357809034_End-to-end_entity-aware_neural_machine_translation)

## Team Members:
- Byomakesh Panda (M24DS004)
- Rishav Kumar (M24DS012)
- Sugandh Kumar (M24DS016)
