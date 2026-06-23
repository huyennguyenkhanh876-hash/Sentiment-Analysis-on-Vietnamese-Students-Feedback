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

1. **Multi-task Bi-LSTM Encoder:** A shared Bidirectional LSTM layer learns representations for both Sentiment and Topic simultaneously, leveraging the semantic correlation between them. The combined loss function is defined as `Loss = 0.5 * L_sent + 0.5 * L_topic`.
2. **Masked Attention Mechanism (Novelty):** Upgraded the standard Attention module by applying a mask (`masked_fill(mask == 0, -1e9)`). This explicitly filters out `<pad>` tokens, forcing the attention weights to focus 100% on actual meaning-bearing words.
3. **Pre-trained Embeddings & Fine-tuning:** Initialized the embedding layer using FastText (`cc.vi.300.vec` - 300 dimensions) with `requires_grad=True` for fine-tuning.
4. **Class-Weighted Loss:** Addressed the severe data imbalance by applying `compute_class_weight('balanced')` from `scikit-learn`. This heavily penalizes the model for misclassifying minority classes like *Neutral* and *Others*.
5. **Vietnamese NLP Preprocessing:** Utilized `pyvi` (`ViTokenizer`) for accurate Vietnamese word segmentation and mapped raw emoticons (e.g., `:)`, `<3`) into text formats to preserve high-value sentiment signals.

## Performance on Test Set
The model's performance is measured using Precision, Recall, and Macro F1-score.
* **Sentiment Analysis:** Achieved a **Macro F1-score of 74.80%**.
* **Topic Classification:** Achieved a **Macro F1-score of 73.80%**.

**Minority Class Breakthrough:** 
Despite the overall F1-score being lower than the MaxEnt baseline, the proposed techniques yielded impressive results on the most difficult minority classes. The *Facility* class (only 3.48% of the data) achieved an outstanding **88.03% F1-score**. The *Neutral* sentiment class reached a **41.62% F1-score**, which is a +7.63% improvement compared to the original MaxEnt baseline (33.99%).

## Error Analysis & Limitations
* **The "Others" and "Neutral" Bottleneck:** The primary reason the overall Macro F1 trails the baseline (87.94% for Sentiment, 84.03% for Topic) is the poor performance on the *Neutral* and *Others* classes. The *Others* class acts as a semantic "catch-all" without distinct keyword patterns, leading to 18% confusion with *Lecturer* and 20% with *Curriculum*.
* **Overfitting Tendency:** Learning curves indicate validation loss diverging from training loss from Epoch 6 onwards, highlighting a gap in generalization. Early stopping was successfully triggered at Epoch 13 to mitigate this.
* **Vocabulary Limitations:** FastText embeddings only covered ~49% (1008/2069) of the vocabulary. Student-specific slang and abbreviations (e.g., *gv*, *sv*, *ko*) lacked pre-trained representations, resulting in a loss of valuable information.

## Future Improvements
* Replace Bi-LSTM with **PhoBERT** to capture deeper contextual relationships in Vietnamese text.
* Implement Data Augmentation (e.g., Back-translation, Paraphrasing) specifically for *Neutral* and *Others* classes.

