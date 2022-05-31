import pickle
import seaborn as sns
import matplotlib.pyplot as plt
with open(f'trainer/confidences_error_train.pickle', 'rb') as f:
    confidences_error = pickle.load(f)


sns.displot(x=confidences_error)
plt.show()