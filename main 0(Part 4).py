from os import listdir
from os.path import isfile, join
import cv2
import numpy as np

from PIL import Image
from matplotlib import cm
import scipy.io

def get_image_segs_name():
    image_path = "data\\images"
    segs_path = "data\\groundTruth"

    images_names = [f for f in listdir(image_path) if isfile(join(image_path, f))]
    segs_names = [f for f in listdir(segs_path) if isfile(join(segs_path, f))]
    return images_names,segs_names

def image_reader(image_path,segs_path,image_name,seg_name):
    img = cv2.imread(image_path + '\\' + image_name)
    # print(img.shape)
    x = scipy.io.loadmat(segs_path + "\\" + seg_name)
    temp_data = x['groundTruth'][0]

    segments = np.zeros((5, 321 * 481))
    boundries = np.zeros((5, 321 * 481))
    for j in range(5):
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
            temp_centres[j, :] = np.mean(dataset[indexes], axis=0)
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
def calc_confusion_matrix(clusters,ground_truth,k):
    confusion=np.zeros(shape=(k,len(ground_truth)),dtype=int)

    for i in range(len(clusters)):
        for j in range(len(ground_truth)):
            if i in int(ground_truth[j]):
                m=int(clusters[i])
                confusion[m,j]=confusion[m,j]+1
    return confusion

def calc_confusion_matrix2(clustered,ground_truth,k):
    new_shaped_data = np.zeros(shape=(k,len(ground_truth)), dtype=int)
    clusters_sizes = []
    m = 0
    diff_shapes = []
    for i in range(0, k):
        m = 0
        for j in range(len(clustered)):
            if clustered[j] == i:
                new_shaped_data[i, m] = ground_truth[j]
                m += 1
                if ground_truth[j] == 0:
                    print("Zerooooooooooooooooooo")
                if ground_truth[j] not in diff_shapes:
                    diff_shapes.append(ground_truth[j])
        #new_shaped_data[i] = new_shaped_data[i, :m]
        clusters_sizes.append(m)
    return new_shaped_data, diff_shapes, clusters_sizes

def cond_entropy(clustered_ground, diff_shapes, N, clusters_sizes):
    total_entropy = 0
   # table = dict(zip(diff_shapes, np.zeros(len(diff_shapes))))
    from math import log

    for i in range(len(clustered_ground)):
        # Calculating number of each shape in each cluster
        ni = clusters_sizes[i]
        table = dict(zip(diff_shapes, np.zeros(len(diff_shapes))))
        #print(diff_shapes)
        #print(table)
        #print(table.get(0))
        #for k in range(len(diff_shapes)):
        for j in range(ni+1):
            #print(table)
            key = clustered_ground[i, j]
            if(key != 0):
                #print(key)
                num = table.get(key)+1
                new = {key: num}
                table.update(new)
        # Calculating entropy
        entropy = 0
        for m in range(len(diff_shapes)):
            print("m:", m)
            print("Shape: ", diff_shapes[m])
            print(diff_shapes)
            print("table: ", table)
            fraction = (table.get(diff_shapes[m])/ni)
            print("Table's: ", table.get(diff_shapes[m]))
            print("ni: ", ni)
            print("Fraction: ", fraction)
            ent = -fraction*log(fraction, 2)
            print("Ent: ", ent)
            entropy += ent
        total_entropy += (ni/N)*entropy

    return total_entropy

image_path = "data\\images"
segs_path = "data\\groundTruth"
image_names,image_segs=get_image_segs_name() #get all the file names in a directory
# print(image_names)
# print(image_segs)
'''
image,seg,boundries=image_reader(image_path,segs_path,image_names[5],image_segs[5])# your segmantation function + boundries
k=5
#print(image)
#print(seg[0].shape)
#print(seg[0][0:200])
#print(boundries)
b = image.reshape(321 * 481, 3)
#print(b)
#from sklearn.cluster import KMeans

#centres_history, labels_history = kmeans(b, k) # k means implementation
#a = labels_history[-1] # get the final clustring

from sklearn.cluster import KMeans
clustered = KMeans(n_clusters=k, random_state=0)
clustered.fit(b)
#print(clustered)
colors = clustered.cluster_centers_
print(colors)
labels = clustered.labels_
#print(labels.shape)
'''
def plotClusters(LABELS, IMAGE, CENTROID_COLORS):
    new_image = []
    # plotting
    #fig = plt.figure()
    #ax = Axes3D(fig)
    for label, pix in zip(LABELS, IMAGE):
        #ax.scatter(pix[0], pix[1], pix[2], color=rgb_to_hex(COLORS[label]))
        curr_color = CENTROID_COLORS[label]
        new_image.append(curr_color)
        #print(label, pix)
    #plt.show()
    #print(new_image)
    #print(new_image[3])
    new_image = np.reshape(new_image, (321, 481, 3))
    from matplotlib import pyplot as plt
    plt.imshow(new_image, interpolation='nearest')
    plt.show()

#plotClusters(clustered.labels_, b, colors.astype(int))

def process_five_images_KMEANS(k):
    from sklearn.cluster import KMeans
    for i in range(20, 26):
        clustered = KMeans(n_clusters=k, random_state=0)
        image, seg, boundries = image_reader(image_path, segs_path, image_names[i], image_segs[i])
        imageAs1D = image.reshape(321 * 481, 3)
        clustered.fit(imageAs1D)
        colors = clustered.cluster_centers_
        labels = clustered.labels_
        visualize_image_gt(image, seg, boundries)
        plotClusters(clustered.labels_, imageAs1D, colors.astype(int))



def process_five_images_NCUT(k):
    from sklearn.neighbors import kneighbors_graph
    from sklearn.cluster import spectral_clustering
    from sklearn.cluster import SpectralClustering
    from sklearn.cluster import KMeans
    for i in range(21, 26):
        image, seg, boundries = image_reader(image_path, segs_path, image_names[i], image_segs[i])
        imageAs1D = image.reshape(321 * 481, 3)
        #simMatrix1 = kneighbors_graph(imageAs1D, 5)
        #print(np.shape(simMatrix1))
        #simMatrix = simMatrix1.toarray()
        clustered = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=5, assign_labels='kmeans', n_jobs=1).fit(imageAs1D)
        #colors = clustered.cluster_centers_
        #spectral_clustering(simMatrix1, n_clusters=k, n_components=None, assign_labels='kmeans')
        print(clustered.labels_)
        colors =[]
        import random
        for j in range(0, k):
            r = random.randint(0, 255)
            colors.append([r, r, r])
            #print('#%02X%02X%02X' % (r(), r(), r()))

        print(colors)
        visualize_image_gt(image, seg, boundries)
        plotClusters(clustered.labels_, imageAs1D, colors)



process_five_images_NCUT(5)

#process_five_images_KMEANS(5)




#TRY FOR N CUT
'''
# Delta matrix ( Degree matrix )
def buildDegreeMatrix(simMatrix):
    diag = np.array(simMatrix.sum(axis=1)).ravel()
    result = np.diag(diag)
    return result
# La Matrix ( Difference between degree and similarity matrix
def buildLaplacianMatrix(simMatrix, degreeM):
    result = degreeM - simMatrix
    return result

def process_five_images_NCUT(k):
    from sklearn.neighbors import kneighbors_graph
    from sklearn.cluster import KMeans
    for i in range(20, 26):
        image, seg, boundries = image_reader(image_path, segs_path, image_names[i], image_segs[i])
        imageAs1D = image.reshape(321 * 481, 3)
        simMatrix1 = kneighbors_graph(imageAs1D, 5, mode='connectivity')
        simMatrix = simMatrix1.toarray()
        delaMatrix = buildDegreeMatrix(simMatrix)
        # print(delaMatrix)
        laplacianMatrix = buildLaplacianMatrix(simMatrix, delaMatrix)
        # print(laplacianMatrix)
        eig_values, eigVectors = np.linalg.eigh(laplacianMatrix)
        ind = eig_values.real.argsort()[:k]
        result = np.ndarray(shape=(laplacianMatrix.shape[0], 0))
        for i in range(0, ind.shape[0]):
            egVecs = np.transpose(np.matrix(eigVectors[:, np.asscalar(ind[i])]))
            result = np.concatenate((result, egVecs), axis=1)

        clustered = KMeans(n_clusters=k, random_state=0)
        clustered.fit(result)
        colors = clustered.cluster_centers_

        plotClusters(clustered.labels_, imageAs1D, colors.astype(int))
        visualize_image_gt(image, seg, boundries)

'''

'''new_image = np.reshape(plotClusters(clustered.labels_, b, colors.astype(int)), (321, 481, 3))
from matplotlib import pyplot as plt

plt.imshow(new_image, interpolation='nearest')
plt.show()'''

#print(zip(labels, b))


# print(labels_history)
#print(a.shape)
#print(a[0:200])
#output, diff_values ,clusters_sizes= calc_confusion_matrix2(a, seg[0], k)
#print(clusters_sizes)
#print(diff_values)

'''for i in range(0,3):
    for j in range(len(output[0])):
        if(output[i, j] == 0):
            print("ZEROOOOOOOOOO Damn!")
'''
#cond_entroppy = cond_entropy(output, diff_values, len(a), clusters_sizes)

#print(cond_entroppy)
'''for i in range(0, k):
    print("Class number :", i)
    for j in range(0, 201):
        print(out[i, j])
'''

#print(calc_confusion_matrix2(a, seg[0], k)[0][0:200])
#print(calc_confusion_matrix2(a, seg[0], k)[1][0:200])
#print(calc_confusion_matrix2(a, seg[0], k)[2][0:200])

#from sklearn.metrics.cluster import completeness_score
#print(completeness_score(seg[0], a))
#k,b,c=image.shape
#f=a.reshape((k,b)) # return the image to it's normal dimensions
# dataset ,segmantes,boundries,shapes=dataset_reader()
#visualize_image_gt(image,seg,boundries) # your function + boundries + segmentation
from matplotlib import pyplot as plt
# segTry=segmantes[0]
# boundTry=boundries[0]
# a,b,c=shapes[0]
# trial=boundTry.reshape(5,a,b)

#plt.imshow(f, interpolation='nearest')
#plt.show()


