import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# vader_lexicon 다운로드
nltk.download('vader_lexicon')

# SentimentIntensityAnalyzer 인스턴스 생성
sia = SentimentIntensityAnalyzer()

# 분석할 텍스트
text = "I really enjoyed the new movie. The acting was great and the plot was engaging."

# 감정 분석 수행
scores = sia.polarity_scores(text)

# 결과 출력
print(scores)
