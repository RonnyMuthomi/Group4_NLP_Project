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


## Modeling

For the modeling phase, the processed_tweet column containing cleaned and preprocessed tweet text served as the main input (X), and the sentiment column was used as the target variable (y). To begin, a baseline Logistic Regression model was implemented due to its simplicity, interpretability, and strong performance in text classification tasks. The text was first vectorized using TF-IDF, which transformed each tweet into a numerical feature vector based on the importance of terms across the corpus.

To improve model performance, hyperparameter tuning was performed using GridSearchCV. This allowed systematic exploration of parameters such as regularization strength (C) and solver options to reduce overfitting and improve generalization. Once tuned, the Logistic Regression model was evaluated and showed good accuracy, especially on majority classes.

To benchmark performance, additional models such as Naive Bayes, Support Vector Machines (SVM), and Random Forest Classifiers were tested. Each model was evaluated using accuracy, precision, recall, and F1-score, with special focus on weighted F1-score due to class imbalance. Among them, Logistic Regression and SVM showed the most consistent and reliable results on both training and test data.

## Evaluation

Model evaluation focused on both overall performance and how well minority sentiment classes were captured. Given the imbalance in sentiment labels, weighted F1-score was chosen as the primary evaluation metric, since it balances precision and recall across all classes proportionally. The confusion matrix revealed that most misclassifications occurred between neutral and positive classes, which often share similar vocabulary, making them harder to distinguish.

While the accuracy metric gave a general sense of correctness, deeper insights came from looking at per-class metrics. The model achieved higher precision and recall for the "Positive" and "No emotion" classes, but struggled slightly with the "Negative" and "I can't tell" sentiments. These results were expected, given the limited representation of those classes in the dataset.

## Hyperparameter Tuning

To further refine the model, GridSearchCV was used to search over a range of parameters. For the Logistic Regression model, key parameters such as the regularization strength (C) and penalty (l2) were explored. Additionally, TF-IDF vectorizer parameters like ngram_range, min_df, and max_df were tuned to find the most informative set of textual features. These changes led to modest but meaningful improvements in performance, especially in balancing recall across all sentiment categories.

The grid search process ensured that the final model was not only accurate but also generalizable, avoiding overfitting to training data and maintaining strong performance on unseen tweets.

## Final Model Selection

After evaluating multiple models and tuning their parameters, Logistic Regression with TF-IDF vectorization was selected as the final model. It achieved the best balance between interpretability, computational efficiency, and predictive power. The model handled the imbalanced classes reasonably well, especially after applying weighting strategies and careful feature engineering.

This final model was capable of accurately classifying tweet sentiment based on the content of the text, fulfilling the primary objective of the project.

## Conclusion

This project successfully demonstrated the application of Natural Language Processing (NLP) techniques to classify tweet sentiment using machine learning. Starting with raw, noisy Twitter data, the pipeline involved systematic text preprocessing cleaning, tokenization, stop word removal, stemming and transforming the text into numerical vectors via TF-IDF. Through rigorous modeling and hyperparameter tuning, the best-performing model was selected based on weighted F1-score, ensuring fair evaluation across all sentiment classes.

The final model can now predict whether a tweet expresses positive, negative, neutral, or ambiguous emotion toward a product or brand, providing valuable insights for businesses, marketers, or researchers analyzing public sentiment. Future improvements may include integrating deep learning techniques or using contextual embeddings like BERT for even better performance on subtle linguistic cues.


