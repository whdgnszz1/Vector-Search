from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 데이터 세트 로드
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# 텍스트 데이터 벡터화
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(newsgroups_train.data)
X_test = vectorizer.transform(newsgroups_test.data)

# 나이브 베이즈(Naive Bayes) 분류기 훈련
clf = MultinomialNB()
clf.fit(X_train, newsgroups_train.target)

# 테스트 세트 예측
y_pred = clf.predict(X_test)

# 정확도와 예측한 클래스 래이블 출력
print(f"Accuracy: {accuracy_score(newsgroups_test.target, y_pred)}")
print(f"Predicted classes: {y_pred}")
