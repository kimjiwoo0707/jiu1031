AI 프로젝트1
<졸음운전 방지 시스템>
머신러닝 알고리즘을 이용하여 졸음운전 판별 여부를 나눈다.

프로젝트 개요
졸음 운전으로 인하여 나는 사고와 사망률이 점차 늘어나는 추세이다. 이를 방지하기 위하여 사전에 졸음을 인지하여 큰 사고로 이어지지 않도록 졸음운전 판별 여부를 확인한다.

필요한 라이브러리
numpy
pandas
os
cv2
dlib
train_test_split
StandardScaler
accuracy_score, classification_report
Sequential
Conv2D, Flatten, Dense, BatchNormalization, Dropout
Adam

프로젝트 한계
사람의 얼굴의 변화를 바로바로 알아보지 못하는 경우 졸음 운전 여부를 판별하지 못하는 경우도 있다.
예를 들어 얼굴이 붓거나 짙은 선글라스를 끼는 경우 판별이 어렵다.
