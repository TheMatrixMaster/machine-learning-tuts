# data visualization

import pandas as pd 
import matplotlib.pyplot as plt 

train_csv = pd.read_csv("train.csv", index_col=0)
test_csv = pd.read_csv("test.csv", index_col=0)

classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
colors = ['C0', 'C1', 'C1', 'C1', 'C1']
num_cases = []

for i in range(0, len(classes)):
	num_cases.append(len(train_csv.ix[train_csv['diagnosis'] == i]))
	print("There are %d cases of %s" % (num_cases[i], classes[i]))


plt.bar(classes, num_cases, color=colors)
plt.title("Number of training samples per class for diabetic retinopathy")
plt.show()

classes = ['No DR', 'DR']
colors = ['C0', 'C1']
num_cases = [num_cases[0], sum(num_cases[1:])]

plt.bar(classes, num_cases, color=colors)
plt.title("Distribution of positive and negative cases for dataset")
plt.show()