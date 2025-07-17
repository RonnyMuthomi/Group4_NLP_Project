# Tweet Sentiment Classification

## Predicting Tweet Sentiment Based on Content Using Natural Language Processing (NLP)


## The Problem

<img width="1366" height="331" alt="image" src="https://github.com/user-attachments/assets/86e587f0-b681-4c5b-97bd-091d47e95081" />


Social media platforms like Twitter are powerful mediums for users to express their opinions, emotions, and experiences. However, extracting meaningful insights from thousands of unstructured tweets can be challenging. Brands, governments, and researchers increasingly rely on sentiment analysis to understand public opinion and respond accordingly.
This project aims to build a **machine learning model** that can automatically classify the sentiment of a tweet **Positive, Negative, No Emotion, or I Can’t Tell** based solely on its text content. **The core problem is the unstructured nature of tweet data**, making it necessary to apply NLP preprocessing, vectorization, and robust classification techniques.

By accurately predicting tweet sentiment, the model can support

- Social listening for brands and organizations

- Crisis monitoring and public feedback

- Automated moderation and analytics pipelines

## Business Understanding
In today's digital era, sentiment expressed on platforms like Twitter often influences consumer behavior, brand perception, and even political opinion. Organizations need reliable tools to monitor sentiment trends and automatically analyze public opinion in real time.

<img width="1366" height="331" alt="image" src="https://github.com/user-attachments/assets/8e4f5eab-dd2b-42af-a060-f5d15480492d" />


This project offers a practical solution by using Natural Language Processing (NLP) and machine learning to process raw tweets and classify their sentiment. The resulting model can help stakeholders
- Gauge public reaction to a product, event, or policy

- Detect and address negative feedback early

The key stakeholder here is any organization or analyst interested in understanding human emotion and behavior through digital text.


## Data Understanding

The dataset used in this project consists of over 9,000 real-world tweets, each annotated with three key columns: tweet_text, product, and sentiment. The tweet_text column contains the raw content of the tweet, which serves as the primary input for natural language processing tasks. The product column identifies the brand or item referenced in the tweet ("iPhone"), while the sentiment column is the target variable representing the emotion expressed. Sentiment labels include Positive emotion, Negative emotion, No emotion toward brand or product, and I can’t tell. The dataset, sourced from [CrowdFlower via data.world](https://data.world/crowdflower/brands-and-product-emotions)

A preliminary review of the data revealed that the tweet text is often noisy and contains elements such as user mentions, hashtags, and URLs, all of which can interfere with model training if not properly cleaned. Furthermore, the dataset exhibits class imbalance, with a greater proportion of tweets expressing neutral or positive sentiment compared to negative ones. Since the core objective is to classify sentiment based on tweet content, most preprocessing and feature engineering steps are focused on the tweet_text column.

## Data Preparation

Preprocessing played a critical role in transforming raw tweet text into a form suitable for machine learning. The process began with text cleaning, where regular expressions (regex) were used to remove URLs, user mentions, hashtags, punctuation, and to convert all text to lowercase for consistency. Next, the cleaned text was tokenized using NLTK, breaking each tweet into individual words or tokens. To enhance relevance, stop words common words like "the" and "is" that do not carry meaningful sentiment were removed. The tokens were then passed through a stemming process, reducing words to their root form ("running" to "run"), which helped reduce dimensionality. Finally, TF-IDF vectorization was applied to convert the processed text into numerical feature vectors, making it possible to train machine learning models. These steps ensured the model could focus on the most informative features, and the cleaned, tokenized, and vectorized versions of each tweet were stored in a new column named processed_tweet.

<img width="746" height="72" alt="image" src="https://github.com/user-attachments/assets/16013899-29a6-4e9f-86c1-d353fc8c043a" />

## Exploratory Data Analysis (EDA)

The goal of Exploratory Data Analysis EDA in this project was to gain insights into the structure and patterns within the dataset to inform better modeling decisions. One of the first observations was the clear class imbalance in the sentiment labels most tweets reflected either no emotion or positive sentiment, while negative sentiments were significantly fewer. This imbalance highlighted the need for careful evaluation metrics and possibly rebalancing techniques during model training.

<img width="1366" height="500" alt="image" src="https://github.com/user-attachments/assets/681d2247-031b-458a-a10d-97647c665fa4" />


Further inspection of the tweets by product category showed how sentiment varied across different brands, suggesting that brand perception might influence sentiment expression. Visualizations like bar plots and pie charts were used to show the frequency distribution of sentiments and products. Additionally, text-specific analyses such as word clouds and most frequent terms revealed common themes and vocabulary associated with each sentiment class. For example, negative tweets often contained words like "broken," "hate," or "problem," while positive tweets featured terms like "love," "great," or "happy." These findings confirmed that the textual content carried distinguishable sentiment signals that a model could learn from.

<img width="1366" height="547" alt="image" src="https://github.com/user-attachments/assets/8c6d5cfa-823f-4979-82b0-6ddaeee9304c" />

---
## Modeling

For the modeling phase, we used the `processed_tweet` column containing cleaned and preprocessed tweet text as input features (X), and the `sentiment` column as the target variable (y).  
As a starting point, we implemented a baseline **Logistic Regression** model because of its simplicity, interpretability, and proven effectiveness in text classification.

The text data was vectorized using **TF-IDF** (Term Frequency–Inverse Document Frequency), transforming each tweet into a numerical feature vector that captures the importance of words across the corpus.

---

###  Benchmarking Multiple Models

To benchmark performance, we trained and compared several machine learning models:
- Logistic Regression
- Random Forest
- Naive Bayes
- K-Nearest Neighbors (KNN)
- XGBoost

Each model was built as a pipeline:
1. TF-IDF vectorizer to transform text
2. Classifier as the final estimator

---

###  Evaluation & Metrics

All models were evaluated on the test set using:
- **Accuracy** – overall correctness
- **Precision, Recall, F1-score** – to balance false positives and false negatives
- **Weighted F1-score** – chosen as the primary metric due to class imbalance (to give fair weight to minority classes)

**Best results:**
| Model                | Accuracy | Weighted F1-score |
|---------------------|:--------:|:-----------------:|
| Random Forest       | ~0.68    | ~0.66             |
| Logistic Regression | ~0.67    | ~0.64             |
| XGBoost             | ~0.66    | ~0.63             |
| Naive Bayes         | ~0.65    | ~0.62             |
| KNN                 | ~0.64    | ~0.59             |

The **Random Forest** pipeline achieved the best overall accuracy (~68%) and balanced performance across sentiment classes.

---

### Hyperparameter Tuning

To boost performance, we performed **GridSearchCV** tuning:
- For Logistic Regression: regularization strength `C`, penalty type, solver
- For Random Forest: number of estimators, max depth, and class weighting
- For KNN: number of neighbors, distance metric
- For Naive Bayes: smoothing parameter `alpha`
- For XGBoost: learning rate, number of estimators, max depth

Tuning led to meaningful improvements, particularly in recall for minority sentiment classes.

##  Key Visuals from Modeling Phase

 **Confusion Matrix**  
<img width="600" alt="Confusion Matrix" src="images/confusion_matrix.png" />

 **Classification Report Heatmap**  
 
<img width="758" height="656" alt="image" src="https://github.com/user-attachments/assets/4178c42f-8db0-463d-adec-27136439e766" />


 **Top Influential Features (Logistic Regression)**  
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/e13c480f-b996-4540-a52f-897256cc8758" />

 **Learning Curve**  
<img width="989" height="790" alt="image" src="https://github.com/user-attachments/assets/9f5e2bab-e5e1-4a8e-87ec-2ff6f4673580" />

---

###  Final Model Selection

After comparing models, **Random Forest with TF-IDF** vectorization was selected as the final model.  
Reasons:
- Best trade-off between accuracy and F1-score
- Robust performance on unseen data
- Ability to handle class imbalance better after tuning

The final model achieved:
- **Accuracy:** ~0.68
- **Weighted F1-score:** ~0.66

---

## Conclusion

By systematically preprocessing text, building pipelines, tuning hyperparameters, and evaluating multiple models, the project successfully built a robust text classifier.  
This model can now predict whether a tweet expresses positive, negative, neutral, or uncertain emotion, helping businesses and analysts gain insights from social media sentiment.

