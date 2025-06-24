# Att-ResRoBERTa: A Multimodal Sarcasm Detection Model for Code-Switched Tagalog-English Content

Att-ResRoBERTa is a multimodal deep learning model developed for sarcasm detection in code-switched Tagalog-English (Taglish) memes. It extends the architecture proposed by Pan, Lin, Fu, Qi, & Wang (2020), leveraging attention mechanisms to capture textual-visual incongruity. This work focuses on the Filipino social media context, where sarcasm is often conveyed through a mixture of Tagalog and English in both text and imagery.

## 🧠 Model Overview

- **Text Encoder**: [XLM-RoBERTa](https://huggingface.co/docs/transformers/model_doc/xlm-roberta) (handles multilingual and code-switched input)
- **Image Encoder**: ResNet152
- **Fusion**: Attention-based fusion layer (adapted from Pan et al., 2020)
- **Classifier**: Fully connected layers for binary classification (sarcastic vs. not sarcastic)

## 🎯 Objectives

- Solve the problem of sarcasm detection in multimodal (text + image) settings.
- Address the challenges of sarcasm in code-switched Tagalog-English memes.
- Evaluate the performance of the proposed Att-ResRoBERTa model.
- Compare results with the BERT-based model from Pan et al. (2020).

## ❓ Statement of the Problem

1. What is the performance of the proposed Att-ResRoBERTa model in detecting sarcasm in Tagalog-English code-switched text-image data?
2. Is there a significant difference between the performance of the BERT-based model (Pan et al., 2020) and Att-ResRoBERTa?

## 📊 Hypothesis

- **Null Hypothesis (H₀)**: There is no significant difference between the performance of the BERT-based model by Pan et al. (2020) and the proposed Att-ResRoBERTa model in detecting Taglish multimodal sarcasm.

## 📈 Key Findings

- **Accuracy**: 75.60%
- **Precision**: 0.71
- **Recall**: 0.82
- **F1-Score**: 0.76
- **Test Statistic**: 6.39
- **p-value**: 0.01145408

![image](https://github.com/user-attachments/assets/cbe763bc-b8e6-465a-9cbb-14460d37591e)


These results led to the **rejection of the null hypothesis** at the 0.05 significance level, suggesting that Att-ResRoBERTa performs significantly better than the baseline BERT-based model in the given context.

## 📁 Dataset

- **Total Samples**: 27 000 (80% Training, 10% Testing, 10% Validation)
- **Languages**: Code-switched Tagalog-English
- **Labels**: Sarcastic / Not Sarcastic
- **Sources**: Public meme pages and online communities
- **Preprocessing**: 
  - Text: Normalization, emoji and special character handling
  - Images: Resizing, standardization

## 🛠️ Technologies Used

- Python 3.10
- PyTorch
- HuggingFace Transformers
- Torchvision
- OpenCV
- Scikit-learn
- Pandas, NumPy

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/attresroberta.git
cd attresroberta
