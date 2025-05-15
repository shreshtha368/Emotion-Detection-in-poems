Emotion Detection in Poems 🎭📝
🔍 Overview
This project presents a hybrid deep learning model for classifying emotions expressed in poems. By combining BERT-based contextual embeddings with Bidirectional LSTM (BiLSTM), the model achieves high performance in multi-class sentiment classification. It categorizes poetic text into positive, negative, or neutral sentiments based on underlying emotional cues such as anger, joy, sadness, courage, etc.

🧠 This work is part of a research study titled "Improving Sentiment Classification Model Using BERT and BiLSTM: A Hybrid Approach" and has achieved 87% accuracy on a dataset of 716 poems.

📁 Dataset
The dataset contains 716 English poems, each annotated with one of the following emotions:

Positive: joy, love, peace, courage, surprise

Negative: anger, hate, fear, sadness

Neutral: (none explicitly labeled, inferred from context)

The poems are mapped to three sentiment classes:

0: Negative

1: Neutral

2: Positive

🛠️ Technologies & Libraries
Transformers: HuggingFace bert-base-uncased, TFRobertaModel

TensorFlow / Keras: Neural network layers and model training

Sklearn: Label encoding, train-test split, class weighting

Pandas / NumPy: Data preprocessing

🧩 Model Architecture
The architecture follows a hybrid NLP pipeline:

Tokenization using BERT tokenizer (bert-base-uncased)

Embedding through pre-trained BERT

BiLSTM Layer to learn sequential context

Global Max Pooling and fully connected Dense layers

Softmax Output for multi-class classification

python
Copy
Edit
# Simplified structure
bert_embeddings → BiLSTM(128) → GlobalMaxPooling → Dense(128) → Dropout(0.4) → Dense(3, softmax)
🧪 Evaluation
Metrics Used:

Accuracy

Precision

Recall

F1 Score

Model	Accuracy	Precision	F1 Score
RoBERTa	78%	0.79	0.78
BiLSTM	80%	0.81	0.80
BERT+BiLSTM (Ours)	87%	0.88	0.87

✅ The hybrid model outperformed standalone BERT or BiLSTM models.

📌 How to Run
Clone the repo:

bash
Copy
Edit
git clone https://github.com/your-username/emotion-detection-in-poems.git
cd emotion-detection-in-poems
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Add the dataset:
Place your Excel file PERC_mendelly.xlsx in the root directory.

Run the notebook:
Open EMOTION_DETECTION_IN_POEMS_MAIN.ipynb in Jupyter or Colab and run all cells.

🎯 Example
Sample Input Poem:

"A new dawn, a day of hope and love, embracing challenges, yet rising above."

Predicted Sentiment: Positive

📚 Literature & Citations
The model is built upon concepts derived from the following foundational papers:

Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” NAACL-HLT, 2018.

Rahman et al., “RoBERTa-BiLSTM: A Context-Aware Hybrid Model for Sentiment Analysis,” arXiv, 2024.

Li et al., “Sentiment Analysis Using BERT and BiLSTM: A Comparative Study,” Journal of Computational Linguistics, 2021.

Wang et al., “A Hybrid Sentiment Classification Model Based on Attention Mechanism,” Neural Networks, 2022.

📄 Full paper: Read Research Paper



