# Natural Language Processing

## Prerequisites
- *Python*: Data structures, NumPy, Pandas, Matplotlib, Seaborn
- *Math*: Basic statistics and linear algebra (Optional)
- *Machine Learning*: Data preprocessing, train-test split, classification, regression, evaluation metrics, overfitting, underfitting

## LEVEL-1: ABCD of NLP

| Day | Topic | Description | Resources |
|-----|-------|-------------|-----------|
| *Day 1* | *Introduction to NLP* | History, applications, NLU and NLG | |
| | *Overview of the full NLP lifecycle* | Text preprocessing<br>Text Representation<br>NLP tasks (classification)<br>Evaluation Metrics (Accuracy, Precision, Recall, F1-score, Confusion Matrix) | |
| | *Tools and libraries Introduction* | -NLTK, spaCy, Scikit-learn<br>- Tokenization, POS tagging, stemming, lemmatization **<br>- Feature extraction pipelines ** | |
| *Day 2* | *Text Preprocessing* | Cleaning & normalizing text data<br><br>i) Tokenization (English vs Bangla) **<br>ii) Stopwords<br>iii) Lemmatization & Stemming<br>iv) Handling punctuations, emojis, special symbols<br>v) Bangla tokenization challenges **| |
| | *Text Representation* | i) Bag of words(BoW)<br>ii) TF-IDF<br>iii) N-grams<br>iv) Intro to embeddings: Word2Vec, GloVe | |
| | *NLP tasks* | Sentiment analysis<br><br>Spam detection | |
| *Day 3* | *NLP Exam-001 (onsite)* | Quiz + Code | |
| | *NLP Project-001 (7 days)* | ## Project topic and guideline ##<br>-Text Classification Project (Classical ML)<br>-Dataset suggestion: Twitter / Bangla sentiments text<br>-Deliverables: preprocessing, TF-IDF, ML model, metrics | |

## LEVEL-2: Context Matters (Deep Learning in NLP)

| Day | Topic | Description | Resources |
|-----|-------|-------------|-----------|
| *Day 1* | *Introduction to Deep Learning* | -Why deep learning?<br>-Feed-forward networks vs sequence modeling. <br>-Limitations of classical ML on text. **| |
| | *Word Embeddings* | -Word2Vec (CBOW, Skip-gram GloVe<br>-FastText (especially good for Bangla subword modeling)<br>-Visualization using PCA/t-SNE | |
| | *Tools and libraries* | Tensorflow / PyTorch, Keras, gensim | |
| *Day 2* | *Sequence Models* | i) RNN<br>ii) LSTM+BiLSTM<br>iii) GRU<br>iv) Sequence padding, masking | |
| *Day 3* | *Application* | i) Sentiment analysis using LSTM<br>ii) Named entity recognition(NER)<br>iii) Machine translation (simple seq2seq) | |
| *Day 4* | *NLP Exam-002 (onsite)* | Quiz + Code | |
| | *NLP Project-002 (7 days)* | ## Project topic and guideline ## Deep Learning NLP Project<br>NER, LSTM sentiment classifier, or simple MT| |

## LEVEL-3: Attention is All You Need (Transformers + LLMs) **

| Day | Topic | Description | Resources |
|-----|-------|-------------|-----------|
| *Day 1* | *Introduction to Transformers* | -Why RNNs fail <br>-Attention mechanism, Self-attention, Multi-head attention<br>-Encoder–decoder overview.| |
| | *Modern NLP Models* | -What is Pretrained model<br>-BERT (encoder), GPT (decoder), T5/BART (encoder-decoder).<br>-Applications: classification, QA, summarization, generation. <br>-Difference between classical NLP vs DL vs LLMs | |t
| *Day 2* | *Transfer Learning in NLP* | - What is fine-tuning?<br>- Task-specific heads<br>- Domain adaptation (medical,finance, Bangla text)<br>- Choosing checkpoints | |
| | *Fine tuning Techniques* | - PEFT (Parameter Efficient Fine Tuning)<br>- LoRA (Low Rank Adaptation)<br>- QLoRA (Quantization + LoRA)<br>- When to use full finetuning vs PEFT| |
| *Day 3* | *Application with Transformers* | - Summarization using T5<br>- Q&A using BERT/GPT/BERT-large<br>- Text generation using GPT-2/smaller models<br>- Embeddings using Sentence-BERT | |
| *Day 4* | *LLMs Now and Future* | - Safety, hallucinations, biases<br>- Where LLMs struggle: reasoning, data gaps| |