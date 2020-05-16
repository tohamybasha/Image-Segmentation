from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from PIL import Image
from matplotlib import cm
import scipy.io

def get_image_segs_name():
    image_path = "ours\\test\\3"
    segs_path = "groundTruth\\test"

    images_names = [f for f in listdir(image_path) if isfile(join(image_path, f))]
    segs_names = [f for f in listdir(segs_path) if isfile(join(segs_path, f))]
    return images_names,segs_names
def image_reader(image_path,segs_path,image_name,seg_name):
    img = cv2.imread(image_path + '\\' + image_name)
    # img = plt.imread(image_path + '\\' + image_name)

    # print(img.shape)
    x = scipy.io.loadmat(segs_path + "\\" + seg_name)
    temp_data = x['groundTruth'][0]

    segments = np.zeros((5, 321 * 481))
    boundries = np.zeros((5, 321 * 481))
    for j in range(3):
        temp_segemant = temp_data[j]['Segmentation'][0, 0]
        # print(temp_segemant.shape)
        temp_segemant = temp_segemant.reshape(321 * 481)
        segments[j, :] = temp_segemant
        temp_boundries = temp_data[j]['Boundaries'][0, 0]
        temp_boundries = temp_boundries.reshape(321 * 481)
        boundries[j, :] = temp_boundries

    return img,segments,boundries
def calc_distance_to_centres(points , centers ):
    distance_matrix=np.zeros(shape=(len(points),len(centers)))
    for  i in range(len(points)):
        for j in range(len(centers)):
            distance_matrix[i,j]=np.linalg.norm(points[i]-centers[j])
    return distance_matrix
def euclidian(a, b):
    return np.linalg.norm(a-b)

def kmeans(dataset, k):
    old_centres= dataset[np.random.randint(1, len(dataset)-1, size=k)]
    new_centres = np.zeros(shape=old_centres.shape)
    distance_between_centres = euclidian(new_centres, old_centres)
    centres_history = []
    labels = np.zeros(shape=(len(dataset)))
    labels_history = []
    while distance_between_centres > 0:

        dis = calc_distance_to_centres(dataset, old_centres)

        for i in range(len(dis)):
            arr1inds = dis[i].argsort()
            # print(arr1inds[0])
            labels[i] = arr1inds[0]
            # print(dataset[i], "---->", dis[i], "--->", labels[i])
        temp_centres = np.zeros((old_centres.shape))
        for j in range(len(old_centres)):
            indexes = [k for k in range(len(dataset)) if labels[k] == j]
            temp_centres[j, :] = np.nanmean(dataset[indexes], axis=0)
        centres_history.append(temp_centres)
        new_centres = temp_centres

        # print(new_centres)
        # print(old_centres)
        distance_between_centres = euclidian(new_centres, old_centres)
        old_centres = new_centres

        labels_history.append(labels)
    # print(labels_history)
    print("centres history")
    return centres_history,labels_history


def visualize_image_gt(image,segmantes,boundries):
    gt = np.zeros(0)

    from matplotlib import pyplot as plt
    # a, b, c = shapes[i]
    #visualize image

    plt.imshow(image, interpolation='nearest')
    plt.show()
    a,b,c=image.shape

    # trial = boundTry.reshape(image.shape)
    k=np.hstack((segmantes[0].reshape((a,b)),segmantes[1].reshape((a,b)),segmantes[2].reshape((a,b)),segmantes[3].reshape((a,b))))

    plt.imshow(k, interpolation='nearest')
    plt.show()

    #visualize segmentics
    k=segmantes[3]
    a,b,c=image.shape
    # f=k.reshape((a,b))
    f=np.hstack((boundries[0].reshape((a,b)),boundries[1].reshape((a,b)),boundries[2].reshape((a,b)),boundries[3].reshape((a,b))))

    from matplotlib import pyplot as plt
    plt.imshow(f, interpolation='nearest')
    plt.savefig(f)
    plt.show()



    # visualize boundries
    # k=boundries[3]
    # a,b,c=image.shape
    # f=k.reshape((a,b))
    # from matplotlib import pyplot as plt
    # plt.imshow(f, interpolation='nearest')
    # plt.show()
    '''for i in range(len(gt)):
        for j in range(len(gt[0])):
            if gt[i][j] == 0:
                gt[i][j] = 255
                '''
    # 1st method for image visualization
    # img1 = Image.fromarray(np.uint8(cm.gist_earth(gt0)*255))
    # img2 = Image.fromarray(np.uint8(cm.gist_earth(gt)*255))
    #
    # img1.show()
    # img2.show()

    # 2nd method for image visualization

    # from matplotlib import pyplot as plt
    # plt.imshow(gt, interpolation='nearest')

    # plt.imshow(gt0, interpolation='nearest')
    #
    # plt.show()
    return
def clac_degre_matrix(sim_matrix):
    rows,columns=np.shape(sim_matrix)
    degree_matrix=np.zeros(shape=(rows,columns))

    for i in range (rows):
        for j in range (columns):
            degree_matrix[i,i]=degree_matrix[i,i]+sim_matrix[i,j]
    return degree_matrix
def calc_normalized_matrix(L_matrix):
    rows, columns = np.shape(L_matrix.T)
    normalized_matrix = np.zeros(shape=(rows, columns))
    # print(normalized_matrix.shape)
    for i in range(rows):
        for j in range(columns):
            a=sum(L_matrix[j,:])
            normalized_matrix[i]=L_matrix[j,i]/a

    return normalized_matrix
def RBF_Kernel(points,gamma):
    sim_matrix =np.zeros(shape=(len(points),len(points)))
    # print(len(points))
    for i in range (len(points)):
        for j in range (len(points)):
            temp=np.linalg.norm((points[i]-points[j])**2)
            temp=np.exp(-gamma*temp)
            sim_matrix[i,j]=temp
    return  sim_matrix
def calc_distance_matrix(points):
    dist_matrix=np.zeros(shape=(len(points),len(points)))
    for i in range (len(points)):
        for j in range (len(points)):
            dist_matrix[i,j]=np.linalg.norm(points[i]-points[j])
    return  dist_matrix
def nn_graph(points,n):
    distance=calc_distance_matrix(points)
    sim=np.zeros(shape=distance.shape)
    for i in range (len(distance)):
        arr1inds = distance[i].argsort()
        sim[i,arr1inds[1:n]]=1
    return sim
def k_normallized_cut(points,k,gamma,similarity_measure=RBF_Kernel):
    sim_matrix = similarity_measure(points, gamma)
    degree_matrix = clac_degre_matrix(sim_matrix)
    L = degree_matrix - sim_matrix
    degree_inv=np.linalg.inv(degree_matrix)
    L_A=degree_inv.dot(L)
    a, b = np.linalg.eig(L_A)
    eigen_values = a[0:k]
    eige_vector = b[0:k]
    norm = calc_normalized_matrix(eige_vector)
    # print(norm)
    centrs,labels = kmeans(points,n_clusters=k)

    c = labels[-1]
    return c

def F_measure(confusion_matrix):
    clusters, ground_truth = confusion_matrix.shape
    Ti = np.zeros(max(ground_truth,clusters))
    ni = np.zeros(max(ground_truth,clusters))
    f = np.zeros(max(ground_truth,clusters))
    print(confusion_matrix.shape)
    for i in range(clusters):
        # Ti[i] = np.sum(confusion_matrix[:, i])
        ni[i] = np.sum(confusion_matrix[i])
        print(i)
        print(np.argmax(confusion_matrix[i]))
        # print(np.argmax(confusion_matrix[:, i]))
        most_expected = confusion_matrix[i, np.argmax(confusion_matrix[i])]
        # expected_true = confusion_matrix[i, np.argmax(confusion_matrix[:, i])]
        purity = most_expected / ni[i]
        # recall = expected_true / Ti[i]
        # print(recall)
        # print(purity)

    for i in range(ground_truth):
        Ti[i] = np.sum(confusion_matrix[:, i])
        print(np.argmax(confusion_matrix[:, i]))
        expected_true = confusion_matrix[np.argmax(confusion_matrix[:, i]), i]
        recall = expected_true / Ti[i]





        f[i] = (2 * purity * recall) / (purity + recall)


    f=sum(f)/clusters
    return f
def conditonal_entropy(confusion_matrix):
    from math import log
    clusters, ground_truth = confusion_matrix.shape
    ni = np.zeros(clusters)
    entropy=np.zeros(clusters)
    f = np.zeros(clusters)
    final_entropy=0

    for i in range(clusters):
        ni[i] = np.sum(confusion_matrix[i])
        # print(ni[i] , " ni ni ni ")

        for j in range(ground_truth):
            # print(confusion_matrix[i,j], "confusion matrix")
            a=confusion_matrix[i,j]/ni[i]
            if a!=0:
                entropy[i]+=(-a*log(a,2))
    out=0
    for i in range(clusters):
        a=ni[i]/sum(ni)

        out+=(a*entropy[i])
    return out

def calc_confusion_matrix(clusters,ground_truth):
    confusion=np.zeros(shape=(len(clusters),len(ground_truth)),dtype=int)
    print(confusion.shape)
    print("calcuating confusion matrix ")
    print(len(clusters))
    print(len(ground_truth))
    print()
    for i in range(len(clusters)):
        for j in range(len(ground_truth)):
            # print(clusters[i])
            for m in range (len(clusters[i])):
                # print(clusters[i][m],"clusterrrrr")
                # print(ground_truth[j],"ground truuuuuth")

                if (clusters[i][m]) in ground_truth[j]:
                    confusion[i,j]=confusion[i,j]+1
                    # print("iiii")

    return confusion
def separate_labels(labels,type):
    print("separating labels ")

    iterations=np.unique(labels)
    # print(iterations)
    clusters=[]
    for i in range(len(iterations)):
        clusters.append([])
    # print(clusters)
    for i in range(len(clusters)):
        for j in range(len(labels)):
            if type==1:
                if labels[j] == (i+1):
                    clusters[i].append(j)
            else :
                if labels[j] == (i):
                    clusters[i].append(j)

    return clusters

def cluster_for_image(image,seg,boundries,k):
    b = image.reshape(321 * 481, 3)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(b)
    clusters = kmeans.labels_
    # centres_history, labels_history = kmeans(b, k)  # k means implementation
    # clusters=labels_history[-1]
    print("done clustring ")

    return clusters
def evaluate_image(clusters,ground_truth):
    cluster_list=separate_labels(clusters,0)
    ground_truth_new = separate_labels(ground_truth[0], 1)

    # for i in range(len(ground_truth))
    #     ground_truth_new=separate_labels(ground_truth[0],1)
    # all_ground_truth=[]
    # for i in range(len(ground_truth)):
    #     ground_truth_list=separate_labels(ground_truth[i])
    #     all_ground_truth.append(all_ground_truth)
    # for i in range(len(all_ground_truth)):
    conf=calc_confusion_matrix(cluster_list,ground_truth_new)
    print(conf)
    f_mesure=F_measure(confusion_matrix=conf)
    entropy=conditonal_entropy(confusion_matrix=conf)
    return f_mesure,entropy

def cluster_images(image_names):
    for j in range(len(image_names)):
        image, seg, boundries = image_reader(image_path, segs_path, image_names[j],
                                         image_segs[j])  # your segmantation function + boundries
        cluster_for_image()

        k=np.array([3,5,7,9,11])
        for i in range(len(k)):
            print('Start clustring image ',image_names[j],"for k =",k[i])
            labels = cluster_for_image(image, seg, boundries, k[i])
            from matplotlib import pyplot as plt
            x, y, z = image.shape
            f= labels.reshape((x, y))  # return the image to it's normal dimensions
            path="C:\\Users\\zeyad\\PycharmProjects\\Image Segmantation\\ours\\test\\"+str(k[i])+"\\"+str(image_names[j])
            plt.imsave(path, f)
            print('image ',image_names[j],"for k =",k[i] ,"is saved")

            # plt.imshow(f, interpolation='nearest')
            # plt.show()
def evaluate_images(image_names,segment_names):
    k = np.array([3,5 ,7, 9, 11])
    f_measure=[]
    f_measure.append([])
    f_measure.append([])
    f_measure.append([])
    f_measure.append([])
    f_measure.append([])
    entropy=[]
    entropy.append([])
    entropy.append([])
    entropy.append([])
    entropy.append([])
    entropy.append([])

    for j in range(len(image_names)):

        for i in range(len(k)):
        # image_path1= "C:\\Users\\zeyad\\PycharmProjects\\Image Segmantation\\ours\\test\\"+str(k[i])
            print("for k = ",k[i] ,"the evaluation of the image ",image_segs[j])
            image, seg, boundries = image_reader(image_path, segs_path, image_names[j],
                                         segment_names[j])
            print("start clustring of image ", image_names[j])
            # print(image)
            image=cluster_for_image(image, seg, boundries, k[i])
            print("start evaluation")
            f_temp,entropy_tmp=evaluate_image(image, seg)
            print("f temp =",f_temp)
            print("entropy =",entropy_tmp)

            print("end evaluation")
            f_measure[i].append(f_temp)
            entropy[i].append(entropy_tmp)


    for i in range(len(f_measure)):
        print("for k =",k[i])
        print("f measures =")
        print(f_measure[i])
        print("f entropy =")
        print(entropy[i])
    print("f measure")
    print(f_measure)
    print("entropy ")
    print(entropy)


image_path = "ours\\test\\3"

image_path = "images\\test"
segs_path = "groundTruth\\test"
image_names,image_segs=get_image_segs_name() #get all the file names in a directory



evaluate_images(image_names[0:20],image_segs[0:20])


'''
# print(len(new_clusters))
# print(new_clusters[4])
# print(f)
from matplotlib import pyplot as plt

# trial=boundTry.reshape(5,a,b)
k,b,c=image.shape
f=labels.reshape((k,b)) # return the image to it's normal dimensions
plt.imsave("C:\\Users\\zeyad\\PycharmProjects\\Image Segmantation\\ours\\test\\dataset",f)
plt.imshow(f, interpolation='nearest')
plt.show()

ground_truth=seg[0].reshape(321*481)
# print(ground_truth.shape)
# visualize_image_gt(image,seg,boundries)
# conf=calc_confusion_matrix(a,ground_truth,3)
# print(conf)
k,b,c=image.shape
f=a.reshape((k,b)) # return the image to it's normal dimensions
# dataset ,segmantes,boundries,shapes=dataset_reader()

# visualize_image_gt(image,seg,boundries) # your function + boundries + segmentation
from matplotlib import pyplot as plt
# segTry=segmantes[0]
# boundTry=boundries[0]
# a,b,c=shapes[0]
# trial=boundTry.reshape(5,a,b)
plt.imshow(f, interpolation='nearest')
plt.show()
'''