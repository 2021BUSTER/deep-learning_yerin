import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# 공부 시간 x1과 과외 시간 x2, 성적 y의 리스트 만들기
data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
y = [i[2] for i in data]

# 그래프로 나타내기
ax = plt.axes(projection='3d')
ax.set_xlabel('study_hours')
ax.set_ylabel('private_class')
ax.set_zlabel('Score')
ax.dist = 11
ax.scatter(x1,x2,y)
plt.show()

# 리스트로 되어 있는 x와 y 값을 넘파이 배열로 바꾸기(인덱스를 주어 하나씩 불러와 계산이 가능하게 하기 위함)
x1_data = np.array(x1)
x2_data = np.array(x2)
y_data = np.array(y)

# 기울기 a와 절편 b의 값, 학습률, 반복횟수 초기화
a1 = 0
a2 = 0
b = 0

lr = 0.02 # 학습률
epochs = 2001 # 몇 번 반복될지 설정

# 경사 하강법 시작
for i in range(epochs):
    y_pred = a1*x1_data + a2*x2_data + b #y = a1*x1 + a2*x2 + b
    error = y_data - y_pred  # 실제 값 - 예측값 : 오차를 구하는 식
    
    
    a1_diff = -(2/len(x1_data)) * sum(x1_data * (error)) # 오차 함수를 a1으로 미분한 값
    a2_diff = -(2/len(x2_data)) * sum(x2_data * (error)) # 오차 함수를 a2로 미분한 값
    b_diff = -(2/len(y_data)) * sum(error) # 오차 함수를 b로 미분한 값

    a1 = a1 - lr * a1_diff #학습률을 곱해 기존의 a값 업데이트
    a2 = a2 - lr * a2_diff #학습률을 곱해 기존의 a값 업데이트
    b = b - lr * b_diff #학습률을 곱해 기존의 a값 업데이터

    if i % 100 == 0:
        print("epoch=%.f, 기울기1=%.04f, 기울기1=%.04f, 절편=%.04f" %(i,a1,a2,b))


y_pred = a1*x1_data + a2*x2_data + b #y = a1*x1 + a2*x2 + b
error = y_data - y_pred  # 실제 값 - 예측값 : 오차를 구하는 식
for i in range(0, len(y_pred)):
    print("y_data : %.04f, y_pred : %.04f, error : %.04f," %(y_data[i], y_pred[i], error[i]))    