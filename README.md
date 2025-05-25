# Fine-Tuning a Generative AI Model for Dialogue Summarization

This notebook focuses on **fine-tuning a large language model (LLM) from Hugging Face for dialogue summarization**. We'll be using **FLAN-T5**, an instruction-tuned transformer known for its summarization capabilities. This guide explores both **full fine-tuning** and **Parameter-Efficient Fine-Tuning (PEFT)** techniques like **LoRA** to optimize model performance. We'll evaluate the results using **ROUGE metrics** and **human evaluation** for comprehensive insights.

---

## Table of Contents

1.  **Set Up Kernel, Dependencies, Dataset & LLM**
2.  **Perform Full Fine-Tuning**
3.  **Perform Parameter-Efficient Fine-Tuning (PEFT)**

---

## 1. Set Up Kernel, Dependencies, Dataset & LLM

This section covers the initial setup, including configuring the kernel, installing necessary dependencies, and loading the dataset and the pre-trained FLAN-T5 model. We'll also perform a zero-shot inference on test examples to establish a baseline.

---

## 2. Perform Full Fine-Tuning

Here, we'll preprocess the dataset specifically for full fine-tuning. The model will then be trained using this approach. We'll evaluate its performance through both **human evaluation** and quantitative **ROUGE metrics** to assess the quality of the generated summaries.

---

## 3. Perform Parameter-Efficient Fine-Tuning (PEFT)

This section focuses on PEFT. We'll configure a **LoRA-based PEFT model** and train the PEFT adapter for efficient fine-tuning. Finally, we'll compare the results of PEFT with those obtained from full fine-tuning using both **human evaluation** and **ROUGE metrics** to understand the trade-offs.

---

## Installation & Dependencies

To run this notebook, ensure you have the following installed:

* Python 3.x
* Jupyter Notebook
* TensorFlow 2.18.0
* PyTorch 2.5.1
* Hugging Face Transformers (v4.38.2)
* Accelerate 0.28.0
* Evaluate 0.4.0
* ROUGE Score 0.1.2
* PEFT 0.3.0

### Usage Instructions

To get started:

1.  Open the Jupyter Notebook: `jupyter notebook Lab_2_fine_tune_generative_ai_model.ipynb`
2.  Execute each section sequentially:
    * Load the dataset and FLAN-T5 model.
    * Fine-tune the model using both full fine-tuning and PEFT approaches.
    * Compare the performance using the specified evaluation metrics.
    * Analyze the results to understand the trade-offs between full fine-tuning and PEFT.

### Contributions

We welcome contributions! Feel free to improve training efficiency, experiment with different datasets, or optimize hyperparameters. Simply fork this repository and submit a pull request.

### License

This project is open-source and released under the [MIT License](https://opensource.org/licenses/MIT).

