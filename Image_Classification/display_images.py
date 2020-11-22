import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2

def display(test_images, model, features):
    images = []
    for key, value in test_images.items():
        for img in range(len(value)):
            images.append(value[img])
    images_staked = np.vstack(images)
    print(images_staked.shape)
    images = np.reshape(images_staked, (-1, 96, 96, 3))
    probabilities = model.decision_function(features)
    probabilities = np.max(probabilities, axis=1)
    predictions = model.predict(features)
    indices_top5 = defaultdict(list)
    indices_bottom5 = defaultdict(list)
    for i in range(1, 2):
        indices_class = list(np.where(predictions == i)[0])
        prob_class = probabilities[indices_class]
        top5 = sorted(prob_class)[-5:]
        bottom5 = sorted(prob_class)[:5]
        for j in indices_class:
            if probabilities[j] in top5:
                indices_top5[str(i)].append(j)
            if probabilities[j] in bottom5:
                indices_bottom5[str(i)].append(j)
    for key, value in indices_top5.items():
            for j in range(len(value)):
                 plt.figure()
                 img = images[value[j]]
                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                 plt.imshow(img)
                 plt.title('Top 5 ranked as class {}'.format(key))
    for key, value in indices_bottom5.items():
            for j in range(len(value)):
                 plt.figure()
                 img = images[value[j]]
                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                 plt.imshow(img)
                 plt.title('Bottom 5 ranked as class {}'.format(key))