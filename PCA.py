import numpy as np
import pandas as pd
from pca import pca

X = np.load("data.npy")
X = pd.DataFrame(data=X, columns=['Utilization', 'Literacy', 'Improvement', 'Tech0logy Barrier', 'Use of Software', 'Training', 'Management Policy', 'ICT', 'Database', 'Digital Tech0logy Skills'])
# Initialize
model = pca()
# Fit transform
out = model.fit_transform(X)

# Print the top features. The results show that f1 is best, followed by f2 etc
print(out['topfeat'])

model.plot()