import random
import math
import matplotlib.pyplot as plt
import numpy as np # Testing for new observations by generating a normal distribution of cluster points rather than randomly generating.
plt.style.use('seaborn-whitegrid')  # Used to plot the elbow plot.

# Tweakable variables
"""
1. number_of_clusters: Number of clusters for K-Means Clustering. Set any negative value if you want to see results for k = 1
to 10. or you can set a specific value of k to get the variance for only that value of k.
2. standardize: This is a boolean variable to let the user choose if he wants to standarize the data or not.
"""
number_of_clusters = -1
standardize = True
# Setting the seed to some integer. So the results will be reproducable.
random.seed(0)

"""
Function extractData(file): This method extracts the input data (X) and targets(y) from the file (iris.data)
"""
def extractData(filename):
    X = []
    y = []
    with open(filename,'r') as data:
        # Read all the data, line by line and store the feature values in 'X' and target values in 'y'.
        for line in data:
            if line != "\n":
                data_in_current_line = line.split(',')
                X.append([float(i) for i in data_in_current_line[:-1]])   # Data sample excluding the target. I am using list comprehension to convert each data item from string to float.
                y.append(data_in_current_line[-1].replace("\n",""))    # Contains the target name for comparision of accuracy we are getting with K-Means Classifier

    return (X,y)

"""
Function standardScaler(X): This function standardizes the given input X and returns X, mean of X, standard deviation of X.
"""
def standardScaler(X):
    sum = [0]*4  # Since we have 4 features
    for i in X:
        sum = [a+b for a,b in zip(i,sum)]
    mean_X = [each_sum/len(X) for each_sum in sum]

    std_X = [0]*4
    for i in X:
        temp = [(a - b)*(a - b) for a,b in zip(i,mean_X)]
        std_X = [a+b for a, b in zip(temp,std_X)]

    std_X = [math.sqrt(each_std/len(X)) for each_std in std_X]

    for i in range(len(X.copy())):
        X[i] = [((X[i][each_X]-mean_X[each_X])/std_X[each_X]) for each_X in range(len(X[i]))]

    return X, mean_X, std_X






"""
Function predictClasses(X, clusters): Parameters- X (Input Samples), clusters (k cluster centers)
This function calculates the distance between each point in X and each of the cluster and based on the minimum distance
we decide the class.
"""
def predictClasses(X, clusters):
    cluster_distances_from_all_points = []      # Example: [[1,3,2],[1,0.4,5]]. Means data point 1 has distances of 1 from cluster1, 3 from cluster2, 2 from cluster3. And point 2 has distances 1,0.4,5 from clusters 1,2,3.
    for each_data_point in X:
        cluster_distances_from_each_point = []
        for each_cluster_point in clusters:
            sum = 0
            # Distance formula calculation
            for i in range(len(each_data_point)):
                temp = each_data_point[i]-each_cluster_point[i]
                temp *= temp
                sum += temp
            cluster_distances_from_each_point.append(math.sqrt(sum))
        cluster_distances_from_all_points.append(cluster_distances_from_each_point)


    # Finding the nearest cluster for each point
    predicted_classes = []
    # The cluster which is nearest to that point, will be the cluster that point belongs to.
    for each_distance in cluster_distances_from_all_points:
        predicted_classes.append(each_distance.index(min(each_distance)))
    return predicted_classes


"""
Function calculateDistortionScore(X,clusters): This function calculates the distortion score (Mean squares) of each point with each cluster center and returns the distortion score.
"""
def calculateDistortionScore(X,clusters):
        predictions = predictClasses(X, clusters)
        cluster_variances = []
        for cluster_number in range(len(clusters)):
            distance = 0
            number_of_labels = 0
            for i in range(len(predictions)):
                if (predictions[i]==cluster_number):
                    # Means X[i] belongs to the cluster_number. So, we calculate the distance square between cluster center and the X[i]
                    number_of_labels+=1
                    for a,b in zip(X[i], clusters[cluster_number]):
                        distance = distance + ((a-b)*(a-b))
            cluster_variances.append(distance)

        total_variance = 0
        for each_variance in cluster_variances:
            total_variance+=each_variance



        return (total_variance/len(X))

"""
Function kmeans: This function creates the new clusters centers finds the points close to it, calculates
the new centers for the clusters by using distance of each point from each cluster.
"""
def kmeans(X,mean_X,std_X,y):
        # Initializing cluster variances
        cluster_variances_list = []

        # Initializing cluster iterations count
        number_of_iterations_to_converge_for_various_k = []

        # Logic to check user input (Based on Tweakable variables provided)
        if number_of_clusters < 0:
            cluster_list = range(1,11)
        else:
            cluster_list = [number_of_clusters]


        for k in cluster_list:   # Why 10, you ask? Its because in most cases we dont like to classify things into more than 10 clusters. And this can also be seen from the distortion score for various K values between 1-10
            clusters = []
            number_of_iterations_to_converge = 0

            for _ in range(k):
                # Since we are in a 4D feature space. Each cluster starting point will be a 4D vector.
                if (standardize):
                    clusters.append([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)])
                else:
                    clusters.append([random.uniform(0,7),random.uniform(0,5),random.uniform(0,8),random.uniform(0,4)])


            all_cluster_centers = []
            while True: # Infinite loop in which we try to converge
                number_of_iterations_to_converge += 1
                predicted_classes = predictClasses(X, clusters)
                # Re-calculating the new centers for each cluster
                new_centers_for_all_clusters = []
                for cluster_number in range(k):     # K is the total number of clusters ("K"-means clustering)
                    new_centers_for_each_cluster = [0]*4    # 4 because the number of features are 4.
                    total = 0
                    number_of_labels = 0
                    for i in range(len(predicted_classes)):
                        if cluster_number == predicted_classes[i]:
                            new_centers_for_each_cluster = [a + b for a, b in zip(new_centers_for_each_cluster, X[i])]
                            number_of_labels+=1

                    try:
                        new_centers_for_all_clusters.append([i/number_of_labels for i in new_centers_for_each_cluster])
                    except Exception as e:
                        if (standardize):
                            new_centers_for_all_clusters.append([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)])
                        else:
                            new_centers_for_all_clusters.append([random.uniform(0,7),random.uniform(0,5),random.uniform(0,8),random.uniform(0,4)])



                clusters = new_centers_for_all_clusters # Assigning new centers for clusters
                #print("Assigned new centers to the existing clusters.")

                all_cluster_centers.append(new_centers_for_all_clusters)
                length_of_iterations = len(all_cluster_centers)

                if(length_of_iterations>=2):
                    list1 = all_cluster_centers[length_of_iterations-1]
                    list2 = all_cluster_centers[length_of_iterations-2]
                    if (list1 == list2):
                        # print("Reached stable centers for k = {} ".format(k))
                        number_of_iterations_to_converge_for_various_k.append(number_of_iterations_to_converge)
                        break

                    else:

                        continue
                        # print("Trying to find new centers! ")

            if number_of_clusters == 3:
                # Calculate the error rate if k = 3. Since we already know the labels we can calculate the error rate.

                max_upto_50 = {0:0,1:0,2:0}
                max_upto_100 = {0:0,1:0,2:0}
                max_upto_150 = {0:0,1:0,2:0}
                for i in range(len(predicted_classes)):
                    if i<50:
                        max_upto_50[predicted_classes[i]]+=1
                    elif 50<=i<100:
                        max_upto_100[predicted_classes[i]]+=1
                    elif 100<=i<150:
                        max_upto_150[predicted_classes[i]]+=1

                max_upto_50 = list(max_upto_50.values()).index(max(list(max_upto_50.values())))
                max_upto_100 = list(max_upto_100.values()).index(max(list(max_upto_100.values())))
                max_upto_150 = list(max_upto_150.values()).index(max(list(max_upto_150.values())))

                incorrect = 0
                for i in range(len(predicted_classes)):
                    if i<50:
                        incorrect+= 1 if predicted_classes[i] != max_upto_50 else 0
                    elif 50<=i<100:
                        incorrect+= 1 if predicted_classes[i] != max_upto_100 else 0
                    elif 100<=i<150:
                        incorrect+= 1 if predicted_classes[i] != max_upto_150 else 0

                print("Accuracy = {}%".format(((150-incorrect)/150)*100))
                exit()

            # Calculating the variance for current value of 'k'
            cluster_variances_list.append(calculateDistortionScore(X,clusters))

        print("Finishing calculations and presenting the variances & number of iterations for given values of k(i.e., number of clusters):")

        # If and else blocks to give results on value of k chosen.
        if len(cluster_list) !=1:
            k_val = 1
            for a,b in zip(cluster_variances_list,number_of_iterations_to_converge_for_various_k):
                print("For k={}: Cluster_variance = {}. Number of iterations to converge = {}".format(k_val,a,b))
                k_val+=1

            plt.plot(range(1,11), cluster_variances_list, 'rx-')
            plt.title("Distortion Scores VS Number of clusters")
            plt.xlabel('Number of clusters')
            plt.ylabel('Distortion scores')
            plt.show()
        else:
            # print(predicted_classes)
            print("For k={}: Cluster_variance = {}. Number of iterations to converge = {}".format(k,cluster_variances_list[0],number_of_iterations_to_converge_for_various_k[0]))



"""
Function main(): This function is the starting point for the program.
"""
def main():
    # Extract the data from from iris.data file
    file_to_read = "./iris.data"
    X, y = extractData(file_to_read)

    # Initializing different cluster centers.
    standardized_X, mean_X, std_X  = standardScaler(X)
    if standardize:
        kmeans(standardized_X,mean_X,std_X,y)
    else:
        kmeans(X,mean_X,std_X,y)


if __name__ == '__main__':
    main()
    if (number_of_clusters < 0):
        number_of_clusters = 3
        print("="*100)
        print("Based on the shown elbow plot. k=3 is the best value for k... ")
        print("Results for k=3:")
        main()
