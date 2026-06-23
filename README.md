# Sentiment-Analysis-on-Vietnamese-Students-Feedback
-----
# Project Overview
This project implements a Deep Learning solution to analyze user viewpoints based on the **UIT-VSFC** (Vietnamese Students’ Feedback Corpus) dataset. The system is designed as a **Multi-task Learning** model capable of simultaneously classifying both the sentiment and the topic of student feedback.

-----
## Dataset Description
The model is trained and evaluated on the UIT-VSFC dataset, which poses a significant challenge due to extreme class imbalance:
* **Sentiment (3 classes):** Negative, Neutral (only ~4.32%), Positive.
* **Topic (4 classes):** Lecturer (~71%), Curriculum, Facility, Others.

## Model Architecture & Technical Innovations
To tackle the linguistic characteristics of Vietnamese student feedback and the imbalanced nature of the dataset, several optimizations were proposed:

1. **Multi-task Bi-LSTM Encoder:** A shared Bidirectional LSTM layer learns representations for both Sentiment and Topic simultaneously, leveraging the semantic correlation between them[cite: 7]. The combined loss function is defined as `Loss = 0.5 * L_sent + 0.5 * L_topic`.
2. **Masked Attention Mechanism (Novelty):** Upgraded the standard Attention module by applying a mask (`masked_fill(mask == 0, -1e9)`). This explicitly filters out `<pad>` tokens, forcing the attention weights to focus 100% on actual meaning-bearing words.
3. **Pre-trained Embeddings & Fine-tuning:** Initialized the embedding layer using FastText (`cc.vi.300.vec` - 300 dimensions) with `requires_grad=True` for fine-tuning[cite: 6, 7].
4. **Class-Weighted Loss:** Addressed the severe data imbalance by applying `compute_class_weight('balanced')` from `scikit-learn`[cite: 6, 7]. This heavily penalizes the model for misclassifying minority classes like *Neutral* and *Others*[cite: 6, 7].
5. **Vietnamese NLP Preprocessing:** Utilized `pyvi` (`ViTokenizer`) for accurate Vietnamese word segmentation and mapped raw emoticons (e.g., `:)`, `<3`) into text formats to preserve high-value sentiment signals[cite: 6, 7].

## 🎯 Performance on Test Set
The model's performance is measured using Precision, Recall, and Macro F1-score[cite: 6].
* **Sentiment Analysis:** Achieved a **Macro F1-score of 74.80%**[cite: 6].
* **Topic Classification:** Achieved a **Macro F1-score of 73.80%**[cite: 6].

**Minority Class Breakthrough:** 
Despite the overall F1-score being lower than the MaxEnt baseline, the proposed techniques yielded impressive results on the most difficult minority classes[cite: 6]. The *Facility* class (only 3.48% of the data) achieved an outstanding **88.03% F1-score**[cite: 6]. The *Neutral* sentiment class reached a **41.62% F1-score**, which is a +7.63% improvement compared to the original MaxEnt baseline (33.99%)[cite: 6].

## 📉 Error Analysis & Limitations
* **The "Others" and "Neutral" Bottleneck:** The primary reason the overall Macro F1 trails the baseline (87.94% for Sentiment, 84.03% for Topic) is the poor performance on the *Neutral* and *Others* classes[cite: 6]. The *Others* class acts as a semantic "catch-all" without distinct keyword patterns, leading to 18% confusion with *Lecturer* and 20% with *Curriculum*[cite: 6].
* **Overfitting Tendency:** Learning curves indicate validation loss diverging from training loss from Epoch 6 onwards, highlighting a gap in generalization[cite: 6]. Early stopping was successfully triggered at Epoch 13 to mitigate this[cite: 6].
* **Vocabulary Limitations:** FastText embeddings only covered ~49% (1008/2069) of the vocabulary[cite: 6]. Student-specific slang and abbreviations (e.g., *gv*, *sv*, *ko*) lacked pre-trained representations, resulting in a loss of valuable information[cite: 6].

## 🚀 Future Improvements
* Replace Bi-LSTM with **PhoBERT** to capture deeper contextual relationships in Vietnamese text[cite: 6].
* Implement Data Augmentation (e.g., Back-translation, Paraphrasing) specifically for *Neutral* and *Others* classes[cite: 6].

---
**Team Members:** Trần Nhật Quang, Đinh Tiến Thành, Nguyễn Khánh Huyền, Võ Hữu Đức Chiến[cite: 6]
