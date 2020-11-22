import os
import cv2
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import imageio

def load_images_from_folder(folder):
    images = {}
    list_of_folders = sorted([f for f in os.listdir(folder) if not f.startswith('.')], key=lambda f: f.lower())
    for filename in list_of_folders:
        category = []
        path = folder + "/" + filename
        for cat in os.listdir(path):
            img = cv2.imread(path + "/" + cat, 0)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is not None:
                category.append(img)
        images[filename] = category
    return images


def load_images_from_folder_color(folder):
    images = {}
    list_of_folders = sorted([f for f in os.listdir(folder) if not f.startswith('.')], key=lambda f: f.lower())
    for filename in list_of_folders:
        category = []
        path = folder + "/" + filename
        for cat in os.listdir(path):
            img = cv2.imread(path + "/" + cat, 1)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is not None:
                category.append(img)
        images[filename] = category
    return images

def sift_features(images, thresh_1, thresh_2):
    sift_vectors = {}
    descriptor_list = []
    #sift = cv2.SIFT_create(contrastThreshold=threshold)
    for key, value in images.items():
        features = []

        if key in ['1']:
            sift = cv2.SIFT_create(contrastThreshold=thresh_1)
        elif key in ['3']:
            sift = cv2.SIFT_create(contrastThreshold=thresh_2)
        elif key in ['4', '5']:
            sift = cv2.SIFT_create(contrastThreshold=0.05, edgeThreshold=8)
        elif key in ['2']:
            sift = cv2.SIFT_create(contrastThreshold=0.07, edgeThreshold=6)
        for img in value:
            kp = []
            des = []
            for i in range(3):
                channel = img[:,:,i]
                kp_c, des_c = sift.detectAndCompute(channel, None)
                kp = [*kp, *kp_c]
                if des_c is not None:
                    des = [*des, *des_c]
                else:
                    print(f'desc none, len kp {len(kp)}')
                    print(key)
            des_staked = np.vstack(des)
            des_staked = np.reshape(des_staked, (-1, 128))
            descriptor_list.append(des_staked)
            features.append(des_staked)
        sift_vectors[key] = features
        print(features[0].shape)

    return descriptor_list, sift_vectors


def sample_dic(dictionary):
    dict_copy = dictionary.copy()
    dict_copy_test = dictionary.copy()
    for key, value in dictionary.items():
        train, test = train_test_split(value, test_size=0.5, random_state=43)
        dict_copy[key] = train
        dict_copy_test[key] = test
    return dict_copy, dict_copy_test


def kmeans(k, descriptor_list):
    kmeans = KMeans(n_clusters=k)
    model = kmeans.fit(descriptor_list)
    return model


def run_clustering(stack_of_descriptors, optimal, dict_for_voc):
    # print('Initialising hyperparameter tuning on kmeans')
    # cluster_sizes = [400, 1000, 4000]
    # values = []
    # for cluster in cluster_sizes:
    # model = kmeans(cluster, descriptor_staked)
    # labels = model.labels_
    # values.append(calinski_harabasz_score(descriptor_staked, labels))
    # print('Running kmeans done')
    # frame = pd.DataFrame({'Cluster': [400, 1000, 4000], 'metric': values})
    # optimal = int(frame[frame.metric == frame.metric.min()]['Cluster'])
    # print('Optimal number chosen')
    print('Final training of kmeans')
    final_model = kmeans(optimal, stack_of_descriptors)
    pickle.dump(final_model, open('kmeans_model.pickle', 'wb'))
    print('Kmeans trained')
    # Create visual vocabulary using the rest of the training data points
    print('Creating visual vocabulary')
    visual_vocabulary = defaultdict(list)
    for key, value in dict_for_voc.items():
        for j in range(len(value)):
            for i in range(len(value[j])):
                feature = value[j][i]
                feature = feature.reshape(1, 128)
                predict_cluster = final_model.predict(feature)
                visual_vocabulary[str(predict_cluster[0])].append(feature)
    print('Visual vocabulary created')
    pickle.dump(visual_vocabulary, open('visual_voc.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return visual_vocabulary, final_model


def construct_histogram(dictionary_images_hist, clusters, model):
    features_for_svm = []
    labels_for_svm = []
    print('Start iteration for creating histogram')
    for key, value in dictionary_images_hist.items():
        print(key)
        im_features = np.array([np.zeros(clusters) for i in range(len(value))])
        labs = np.zeros(len(value))
        for j in range(len(value)):
            for i in range(len(value[j])):
                feature = value[j][i]
                feature = feature.reshape(1, 128)
                predict_cluster = model.predict(feature)
                im_features[j][predict_cluster[0]] += 1
                labs[j] = key
        features_for_svm.append(im_features)
        labels_for_svm.append(labs)
    print('End of iteration')
    features_for_svm = np.vstack(features_for_svm)
    labels_for_svm = np.hstack(labels_for_svm)
    print('Features created for SVM training')
    return features_for_svm, labels_for_svm


def train_SVM(features, labels):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    svm = SVC(random_state=42, C=1)
    model_svm = svm.fit(scaled_features, labels)
    print('Model fitted')
    return model_svm


def overall_training(train_images, th_1, th_2, cluster_no):
    # Getting the descriptors from the images
    descriptor_list, sift_vec = sift_features(images = train_images, thresh_1 = th_1, thresh_2 = th_2)
    print('Sift descriptors created')
    # Sampling the descriptors to just take half from each class
    sift_vec_train, sift_vic_test = sample_dic(sift_vec)
    print('Sampling done')
    descriptor_list_sampled = []
    for key, value in sift_vec_train.items():
        for j in range(len(value)):
            descriptor_list_sampled.append(value[j])
    print('List of descriptors sampled done')
    descriptor_staked = np.vstack(descriptor_list_sampled)
    print('Stacking done')
    print('Initiate function for clustering')
    visual_vocabulary, kmeans_model = run_clustering(descriptor_staked, cluster_no, sift_vic_test)
    print('Initiate function for histograms')
    features, labels = construct_histogram(sift_vic_test, cluster_no, kmeans_model)
    print('Initiate function for SVM')
    final_svm_model = train_SVM(features, labels)
    pickle.dump(final_svm_model, open('svm_model.pickle', 'wb'))
    return final_svm_model, features, labels, kmeans_model

def testing_phase(test_images, cluster_model, svm_model, th_1, th_2, cluster_no):
    descriptor_list, sift_vec = sift_features(images = test_images, thresh_1 = th_1, thresh_2 = th_2)
    print('Descriptor created for test images')
    print('Function for histograms')
    features_test, labels_test = construct_histogram(sift_vec, cluster_no, cluster_model)
    print('Histograms for test images done')
    predictions = svm_model.predict(features_test)
    print('Predictions done')
    print('Classification report')
    class_report = classification_report(labels_test, predictions)
    print('Accuracy score')
    acc_score = accuracy_score(labels_test, predictions)*100
    print('Confusion matrix')
    conf_matrix = confusion_matrix(labels_test, predictions)
    print('Average precision score')
    probabilities = svm_model.decision_function(features_test)
    labels_test_bin = label_binarize(labels_test, classes = [1, 2, 3, 4, 5])
    avg_pred = average_precision_score(labels_test_bin, probabilities)
    pickle.dump(features_test, open('features_test.pickle', 'wb'))
    return class_report, acc_score, conf_matrix, avg_pred, features_test, labels_test

def display_images(test_images, model, features):
    images = []
    for key, value in test_images.items():
        for img in range(len(value)):
            images.append(value[img])
    images_staked = np.vstack(images)
    images = np.reshape(images_staked, (-1, 96, 96, 3))
    probabilities = model.decision_function(features)
    probabilities = np.max(probabilities, axis=1)
    predictions = model.predict(features)
    indices_top5 = defaultdict(list)
    indices_bottom5 = defaultdict(list)
    for i in range(1, 6):
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
            #plt.figure()
            img = images[value[j]]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #plt.imshow(img)
           # plt.title('Top 5 ranked as class {}'.format(key))
            #plt.show()
            directory = './top5_rgbsift_400/' + str(key) + '/'
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    pass
            filename = directory + str(j)
            imageio.imsave("%s.png" % filename, img, format="png")

    for key, value in indices_bottom5.items():
        for j in range(len(value)):
            #plt.figure()
            img = images[value[j]]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #plt.imshow(img)
            #plt.title('Bottom 5 ranked as class {}'.format(key))
            #plt.show()
            directory = './bottom5_rgbsift_400/' + str(key) + '/'
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    pass
            filename = directory + str(j)
            imageio.imsave("%s.png" % filename, img, format="png")

def show_hist(lab, ft, cluster):
    for i in range(1, 6):
        class_list = list(np.where(lab == i)[0])
        indices = list(np.where(lab == i)[0])[::len(class_list)-1]
        histograms = ft[indices]
        for j in range(2):
            plt.bar(x=range(0, cluster), height=histograms[j], width=1.5)
            plt.title('Histogram for class {}'.format(i))
            directory = './histograms_rgbsift_400/' + str(i) + '/'
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    pass
            filename = directory + str(j)
            plt.savefig("%s.png" % filename)
            plt.close()

if __name__ == '__main__':
    train_images = load_images_from_folder_color('./img/')
    #test_images = load_images_from_folder('./img_debug_test/')
    test_images_color = load_images_from_folder_color('./img_testing/')
    print('Images are loaded into dictionary')
    svm_model, features, labels, kmeans_model = overall_training(train_images, th_1 = 0.04, th_2 = 0.01, cluster_no = 400)
    show_hist(labels, features, 400)
    class_Report, acc_score, conf_matrix, average_precision, features_Test, labels_Test = testing_phase(test_images_color
                                                                                       ,kmeans_model, svm_model, th_1 = 0.04, th_2 = 0.01, cluster_no=400)
    display_images(test_images_color, svm_model, features_Test)


