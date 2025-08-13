import pandas as pd
import numpy as np
import math

import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# sklearn data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# sklearn decision trees
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

# evaluation
from sklearn.metrics.pairwise import pairwise_distances_argmin

# tree visualizer
from sklearn import tree

# data visualization
import matplotlib.pyplot as plt

from yellowbrick.cluster import KElbowVisualizer # cluster visualizer


# stuff my mother gave me
def pie_cate(df,category):
    count_value = df.groupby([category]).size().reset_index(name='counts') # find frequency of each unique value
    count_value['%count'] = [round(num/len(df)*100,2) for num in list(count_value['counts'])] # get frequenc distribution
    print(count_value)

    value_list = count_value[category].tolist()
    count_list = count_value['counts'].tolist()
    fig = plt.figure(figsize=(8, 4))

    plt.pie(count_list, labels=value_list)
    plt.show()

def data_exploration (df, column):
    if (str(df[column].dtypes) == 'object'or str(df[column].dtypes) == 'category'):
        if df[column].nunique()<10:
            count_value = df.groupby([column]).size().reset_index(name='counts')
            count_value['%count'] = [round(num/len(df)*100,2) for num in list(count_value['counts'])]
            print(count_value)
            value_list = count_value[column].tolist()
            count_list = count_value['counts'].tolist()
            fig = plt.figure(figsize=(8, 4))
            plt.bar(x=value_list, height= count_list)
            plt.xticks(fontsize=12)
            plt.show()
        else:
            print(column + ' has more than 10 unique values')
    else:
        mean = df[column].describe()['mean']
        std = df[column].describe()['std']
        outlier = df[((df[column]-mean)/std >3) | ((df[column]-mean)/std <-3)][column].tolist()
        if len(outlier) > 0:
            print('There are ' + str(len(outlier)) + ' of outliers for ' + column + '.')
            print(outlier)
        else:
            print('There is no outlier of ' + column + '.')

        ## this is to create box plot
        print('----------------------Box plot---------------------')
        df[column].plot.box(title=column, whis =(5,95))
        plt.grid()
        plt.show()

        ## this is to plot interval column distribution by a decile
        min_value = float(df[column].describe()['min'])
        max_value = float(df[column].describe()['max'])
        if df[column].nunique() >= 10:
            para = (max_value - min_value) / 10
            para_list = np.arange(min_value, max_value, para).round(decimals=2).tolist()
            count_table = df.loc[:, [column]]
            for num in para_list:
                count_table.loc[count_table[column] >= num, 'range'] = num
            count_table_sum = count_table.groupby(['range']).size().reset_index(name='counts')
            value_list = count_table_sum['range'].tolist()
            count_list = count_table_sum['counts'].tolist()
            print('----------------------Distribution plot---------------------')
            fig = plt.figure(figsize=(8, 4))
            plt.bar(x=value_list, height=count_list, width=para, tick_label=value_list, align='edge')
            plt.xticks(rotation=40, fontsize=12)
            plt.grid()
            plt.show()

def metadata(df):
  columns_list = list(df.columns.values) # get a list of column names
  type_list = [str(item) for item in list(df.dtypes)] # get data types
  missing_list = [round(float(num),2) for num in list((df.isnull().sum()/len(df)*100))] # find percentage of missing values
  unique_list = [int(nunique) for nunique in list(df.nunique())] # find unique values for each column
  # # return basic stats for interval columns (i.e. not a category or object datatype and more than 10 unique values)
  metadata = pd.DataFrame(columns_list, columns=['column_name'])
  metadata['datatype'] = type_list
  metadata['missing_percent'] = missing_list
  metadata['unique'] = unique_list
  try:
    desc_interval = df[[item for item in columns_list if str(df[item].dtypes) != 'category' and df[item].nunique()>=10 and str(df[item].dtypes) != 'object']].describe().loc[['mean', 'std', 'min','25%', '50%', '75%', 'max']].transpose().reset_index().rename(columns = {'index':'column_name'})
    metadata = pd.merge(metadata, desc_interval, on='column_name', how='left')
  except:
    metadata
  return metadata


from google.colab import drive
drive.mount('/content/drive')


#filename = '/content/drive/MyDrive/Data/AVONET2_eBird!.xlsx'
#df = pd.read_excel(filename, sheet_name='Caprimulgiformes Traits')
#print(df)


filename = '/content/drive/MyDrive/Papers/Data/AVONET2_eBird!.xlsx'
df = pd.read_excel(filename, sheet_name='Caprimulgiformes Traits')
print(df)


pie_cate(df,'Family2')


df.columns


# Look at information about the data
df2=metadata(df)


# filename = 'AVONET2_eBird!.xlsx'
# df = pd.read_excel(filename, sheet_name='Caprimulgiformes Traits')
# print(df)


y = df.iloc[:, 0].to_numpy() # for predictions
X = df.iloc[:, 1:].to_numpy() # for clustering
print(X.shape, y.shape)


scaler = StandardScaler()
X_std_features = scaler.fit_transform(X)
X_std_features[:5]


model = KElbowVisualizer(KMeans(), k=10)
model.fit(X_std_features)
model.show()


k_means_4 = KMeans(init="k-means++",
                   n_clusters=4,
                   n_init=10,
                   max_iter=300,
                   random_state=101)

k_means_4.fit(X_std_features)


# Add labels
# coordinates of cluster center
k_means_4_centroids = k_means_4.cluster_centers_

# assign a cluster label to each data point
k_means_4_labels = pairwise_distances_argmin(X_std_features,
                                             k_means_4_centroids)

print(k_means_4_labels[:5])


# Cluster 3 is an outlier cluster
unique, counts = np.unique(k_means_4_labels, return_counts=True)
print(unique, counts)


Cluster_4_label_df = df.copy()
Cluster_4_label_df['label'] = k_means_4_labels

print(Cluster_4_label_df.head())


#Cluster_4_label_df.to_csv('/content/drive/MyDrive/Papers/Data/Cluster_4_All.csv')


Cluster_4_label_feature_df = Cluster_4_label_df.iloc[:, 1:]
Cluster_4_label_feature_df.head(3)


# use the following code to plot normalized mean plot

# Extract the list of feature columns from the DataFrame
Cluster_4_column_list = list(Cluster_4_label_feature_df.columns)

# Initialize a list to store the normalized mean values for each feature
Cluster_4_plot_list = []

# Iterate through all features (excluding the last 'label' from clustering)
for feature in Cluster_4_column_list[:-1]:
  plot_dic = {} # Dictionary to hold plotting data for one feature
  plot_dic['feature'] = feature

  # Calculate the mean of the feature for each cluster label (0, 1, 2, 4)
  label_table = pd.DataFrame({
      'mean' : Cluster_4_label_feature_df.groupby(['label'])[feature].mean()
                              }).reset_index()

  # Append a row with the overall mean for the feature
  label_table.loc[len(label_table.index)] = ['overall',
                                             Cluster_4_label_feature_df[feature].mean()]

  # Normalize the mean values to range [0, 1] for fair comparison across features
  label_table['normalize'] = (
      (label_table['mean'] - label_table['mean'].min())
      / (label_table['mean'].max() - label_table['mean'].min())
  )

  # Extract normalized values for each cluster and the overall mean, rounded to 3 decimals
  plot_dic['0_norm'] = round(label_table.iloc[0]['normalize'], 3)
  plot_dic['1_norm'] = round(label_table.iloc[1]['normalize'], 3)
  plot_dic['2_norm'] = round(label_table.iloc[2]['normalize'], 3)
  plot_dic['3_norm'] = round(label_table.iloc[3]['normalize'], 3)
  plot_dic['overall_norm'] = round(label_table.iloc[4]['normalize'], 3)

  Cluster_4_plot_list.append(plot_dic)


# See the normalized mean profile table
Cluster_4_plot_table = pd.DataFrame(Cluster_4_plot_list)
Cluster_4_plot_table


from numpy.ma.core import size
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# write the vertical lines
ax.vlines(x=0, ymin=0, ymax=5, color='black',
          alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=1, ymin=0, ymax=5, color='black',
          alpha=0.7, linewidth=1, linestyles='dotted')

y_reversed_index = list(range(len(Cluster_4_plot_table)))
# y_reversed_index.reverse() # mistakes were made...

ax.scatter(x = Cluster_4_plot_table['0_norm'],
           y = y_reversed_index, marker = 's', s=300,
           c='red', label = '0')
ax.scatter(x = Cluster_4_plot_table['1_norm'],
           y = y_reversed_index, marker = 's', s=300,
           c='green', label = '1')
ax.scatter(x = Cluster_4_plot_table['2_norm'],
           y = y_reversed_index, marker = 's', s=300,
           c='blue', label = '2')
ax.scatter(x = Cluster_4_plot_table['3_norm'],
           y = y_reversed_index, marker = 's', s=300,
           c='orange', label = '3')
ax.scatter(x = Cluster_4_plot_table['overall_norm'],
           y = y_reversed_index, marker = 's', s=300,
           c='grey', label = 'overall')

for i in y_reversed_index:
  ax.text(0-0.05, i, Cluster_4_plot_table.iloc[i][0],
            horizontalalignment='right',
            verticalalignment='center',
            fontdict={'size':14})

ax.legend(loc='lower center',
          ncol=10,
          bbox_to_anchor=(0.5, -0.15))
ax.get_yaxis().set_visible(False)


# See the normalized mean profile table
# Cluster 3 is very large bird with large beak, mas, wing, and Tarsus
# But small Kpps distance and Hand-Wing Index
Cluster_4_plot_table = pd.DataFrame(Cluster_4_plot_list)
Cluster_4_plot_table


Cluster_3_label_feature_df = Cluster_4_label_feature_df[Cluster_4_label_feature_df['label'] != 3]
Cluster_3_label_feature_df


# Extract the list of feature columns from the DataFrame
Cluster_3_column_list = list(Cluster_3_label_feature_df.columns)

# Initialize a list to store the normalized mean values for each feature
Cluster_3_plot_list = []

# Iterate through all features (excluding the last 'label' from clustering)
for feature in Cluster_3_column_list[:-1]:
  plot_dic = {} # Dictionary to hold plotting data for one feature
  plot_dic['feature'] = feature

  # Calculate the mean of the feature for each cluster label (0, 1, 2, 4)
  label_table = pd.DataFrame({
      'mean' : Cluster_3_label_feature_df.groupby(['label'])[feature].mean()
                              }).reset_index()

  # Append a row with the overall mean for the feature
  label_table.loc[len(label_table.index)] = ['overall',
                                             Cluster_3_label_feature_df[feature].mean()]

  # Normalize the mean values to range [0, 1] for fair comparison across features
  label_table['normalize'] = (
      (label_table['mean'] - label_table['mean'].min())
      / (label_table['mean'].max() - label_table['mean'].min())
  )

  # Extract normalized values for each cluster and the overall mean, rounded to 3 decimals
  plot_dic['0_norm'] = round(label_table.iloc[0]['normalize'], 3)
  plot_dic['1_norm'] = round(label_table.iloc[1]['normalize'], 3)
  plot_dic['2_norm'] = round(label_table.iloc[2]['normalize'], 3)
  plot_dic['overall_norm'] = round(label_table.iloc[3]['normalize'], 3)

  Cluster_3_plot_list.append(plot_dic)


# See the normalized mean profile table
Cluster_3_plot_table = pd.DataFrame(Cluster_3_plot_list)
Cluster_3_plot_table


from numpy.ma.core import size
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# write the vertical lines
ax.vlines(x=0, ymin=0, ymax=5, color='black',
          alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=1, ymin=0, ymax=5, color='black',
          alpha=0.7, linewidth=1, linestyles='dotted')

y_reversed_index = list(range(len(Cluster_3_plot_table)))
# y_reversed_index.reverse() # mistakes were made...

ax.scatter(x = Cluster_3_plot_table['0_norm'],
           y = y_reversed_index, marker = 's', s=300,
           c='red', label = '0')
ax.scatter(x = Cluster_3_plot_table['1_norm'],
           y = y_reversed_index, marker = 's', s=300,
           c='green', label = '1')
ax.scatter(x = Cluster_3_plot_table['2_norm'],
           y = y_reversed_index, marker = 's', s=300,
           c='blue', label = '2')
ax.scatter(x = Cluster_3_plot_table['overall_norm'],
           y = y_reversed_index, marker = 's', s=300,
           c='grey', label = 'overall')

for i in y_reversed_index:
  ax.text(0-0.05, i, Cluster_3_plot_table.iloc[i][0],
            horizontalalignment='right',
            verticalalignment='center',
            fontdict={'size':14})

ax.legend(loc='lower center',
          ncol=10,
          bbox_to_anchor=(0.5, -0.15))
ax.get_yaxis().set_visible(False)



Cluster_df = df.drop('Family2', axis=1)
Cluster_df.head(3)


# Train a RF classifier with the default
# Overfitting is OK here because we are not worried about generalizability
RF_4_Kmeans = RandomForestClassifier(random_state=101)

RF_4_Kmeans.fit(Cluster_df, k_means_4_labels)


RF_4_Kmeans_Features = RF_4_Kmeans.feature_importances_

RF_Tree_4_Kmeans_Features_List = []

for i,v in enumerate(RF_4_Kmeans_Features):
  Tree_6_feature_dict = {}
  Tree_6_feature_dict['Features'] = Cluster_df.columns[i]
  Tree_6_feature_dict['Importance'] = round(v,3)
  RF_Tree_4_Kmeans_Features_List.append(Tree_6_feature_dict)

Cluster_4_RF_table = pd.DataFrame(RF_Tree_4_Kmeans_Features_List)

Cluster_4_RF_table.sort_values(by='Importance', ascending=False)


Cluster_3_df = Cluster_3_label_feature_df.drop('label', axis=1)
Cluster_3_df.head(3)


y = Cluster_3_label_feature_df.iloc[:, 11].to_numpy()
X = Cluster_3_label_feature_df.iloc[:, 0:10].to_numpy()
print(X.shape, y.shape)


RF_3_Kmeans = RandomForestClassifier(random_state=101)

RF_3_Kmeans.fit(X, y)


RF_3_Kmeans_Features = RF_3_Kmeans.feature_importances_

RF_Tree_3_Kmeans_Features_List = []

for i,v in enumerate(RF_3_Kmeans_Features):
  Tree_3_feature_dict = {}
  Tree_3_feature_dict['Features'] = Cluster_3_df.columns[i]
  Tree_3_feature_dict['Importance'] = round(v,3)
  RF_Tree_3_Kmeans_Features_List.append(Tree_3_feature_dict)

Cluster_3_RF_table = pd.DataFrame(RF_Tree_3_Kmeans_Features_List)

Cluster_3_RF_table.sort_values(by='Importance', ascending=False)


Cluster_4_label_df2 = Cluster_4_label_df.drop('Family2', axis=1)
Cluster_4_label_df2.head(3)


(X_train, X_test,
 y_train, y_test) = train_test_split(Cluster_df,
                                             k_means_4_labels,
                                             stratify=k_means_4_labels,
                                             test_size=0.2,
                                             random_state=101)

print("The length of training set:", len(X_train))

print("The length of testing  set:", len(X_test))


# Different seeds for training data
(X_train, X_test,
 y_train, y_test) = train_test_split(Cluster_df,
                                             k_means_4_labels,
                                             stratify=k_means_4_labels,
                                             test_size=0.2,
                                             random_state=333)

print("The length of training set:", len(X_train))
print("The length of testing  set:", len(X_test))


# train a DT model using entropy and 1 % of training data
# max depth is the number of clusters
Cluster_4_DT = DecisionTreeClassifier(criterion='entropy',
                                  random_state=100,
                                  max_depth=3,
                                  min_samples_leaf=int(len(X_train)*0.01))

Cluster_4_DT = Cluster_4_DT.fit(X_train, y_train)


# train a DT model using entropy and 1 % of training data with different random state for splitting
# max depth is the number of clusters
Cluster_4_DT_2 = DecisionTreeClassifier(criterion='entropy',
                                  random_state=333,
                                  max_depth=4)
                                  # min_samples_leaf=int(len(Cluster_df)*0.01)

Cluster_4_DT_2 = Cluster_4_DT.fit(Cluster_df, k_means_4_labels)


fig = plt.figure(figsize=(12,12))
_ = tree.plot_tree(Cluster_4_DT_2,
                   feature_names=list(Cluster_df.columns),
                   class_names=['Cluster 0', 'Cluster 1','Cluster 2','Cluster 3'],
                   filled = True)


fig = plt.figure(figsize=(12,12))
_ = tree.plot_tree(Cluster_4_DT,
                   feature_names=list(Cluster_df.columns),
                   class_names=['Cluster 0', 'Cluster 1','Cluster 2','Cluster 3'],
                   filled = True)


(X_train, X_test,
 y_train, y_test) = train_test_split(Cluster_3_df,
                                             y,
                                             stratify=y,
                                             test_size=0.2,
                                             random_state=101)

print("The length of training set:", len(X_train))
print("The length of testing  set:", len(X_test))


# train a DT model using entropy and 3 % of training data
# max depth is the number of clusters
Cluster_DT = DecisionTreeClassifier(criterion='entropy',
                                  random_state=100,
                                  max_depth=3,
                                  min_samples_leaf=int(len(X_train)*0.03))

Cluster_DT = Cluster_DT.fit(X_train, y_train)


fig = plt.figure(figsize=(12,6))
_ = tree.plot_tree(Cluster_DT,
                   feature_names=list(Cluster_3_df.columns),
                   class_names=['Cluster 0', 'Cluster 1','Cluster 2'],
                   filled = True)


# train a DT model using entropy and 1 % of training data
# max depth is 4
Cluster_DT_2 = DecisionTreeClassifier(criterion='entropy',
                                  random_state=100,
                                  max_depth=4,
                                  min_samples_leaf=int(len(X_train)*0.01))

Cluster_DT_2 = Cluster_DT_2.fit(X_train, y_train)

fig = plt.figure(figsize=(12,6))
_ = tree.plot_tree(Cluster_DT_2,
                   feature_names=list(Cluster_3_df.columns),
                   class_names=['Cluster 0', 'Cluster 1','Cluster 2'],
                   filled = True)


beak_length = df[['Family2', 'Beak_Length_Culmen', 'Beak_Length_Nares']].copy()
beak_length


y = beak_length.iloc[:, 0].to_numpy()
X = beak_length.iloc[:, 1:].to_numpy()

print(X.shape, y.shape)



k_means_beak_length = KMeans(init="k-means++",
                   n_clusters=5,
                   n_init=10,
                   max_iter=300,
                   random_state=101)

k_means_beak_length.fit(X)


y = df.iloc[:, 0].to_numpy()
X = df.iloc[:, 1:].to_numpy()


model = KElbowVisualizer(KMeans(), k=10)
model.fit(X)
model.show()


n_clusters1 = 5


k_means_everything = KMeans(init="k-means++",
                   n_clusters=n_clusters1,
                   n_init=10,
                   max_iter=300,
                   random_state=101)

k_means_everything.fit(X)


k_means_know = KMeans(init="k-means++",
               n_clusters=7,
               n_init=10,
               max_iter=300,
               random_state=101)

k_means_know.fit(X)


elbow_labels = k_means_everything.labels_
print(elbow_labels)


know_labels = k_means_know.labels_
print(know_labels)


k_m_df = df.copy()


k_m_df['labels'] = k_means_everything.labels_


k_m_df.to_csv('/content/drive/MyDrive/Data/kmeans_5_cluster.csv')


km_df_k = df.copy()


km_df_k['labels'] = k_means_know.labels_


km_df_k.to_csv('/content/drive/MyDrive/Data/kmeans_7_cluster.csv')


data = df.to_numpy()


from time import time

from sklearn import metrics
from sklearn.#eline import make_#eline
from sklearn.preprocessing import StandardScaler


def bench_k_means(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_#eline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print(82 * "_")
print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

kmeans = KMeans(init="k-means++", n_clusters=n_clusters1, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="k-means++", data=data, labels=labels)

kmeans = KMeans(init="random", n_clusters=n_clusters1, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="random", data=data, labels=labels)

pca = PCA(n_components=n_clusters1).fit(data)
kmeans = KMeans(init=pca.components_, n_clusters=n_clusters1, n_init=1)
bench_k_means(kmeans=kmeans, name="PCA-based", data=data, labels=labels)

print(82 * "_")


import matplotlib.pyplot as plt

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]) #just use the columns themselves

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title(
    "K-means clustering on the digits dataset (PCA-reduced data)\n"
    "Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


