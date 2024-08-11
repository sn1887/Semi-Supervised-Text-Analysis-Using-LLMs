# # Train Data Preprocess
#------------------------------------------------------------------------------------------------------------------
from sklearn.datasets import fetch_20newsgroups

# Fetch the dataset
newsgroups = fetch_20newsgroups(subset='all')

import numpy as np, gc, re 
import pandas as pd 

# Inspecting the dataset
#------------------------------------------------------------------------------------------------------------------
print("Categories:", newsgroups.target_names)
print("Number of  samples:", len(newsgroups.data))

train_id = [f'{i:05d}' for i in range(1, len(newsgroups.data) + 1)]

train = pd.DataFrame({'unique_id' : train_id , 'full_text': newsgroups.data, 'labels': newsgroups.target})

# EMBEDDINGS TO LOAD/COMPUTE
# PARAMETERS = (MODEL_NAME, MAX_LENGTH, BATCH_SIZE)
# CHOOSE LARGEST BATCH SIZE WITHOUT MEMORY ERROR

models = [
    ('microsoft/deberta-base', 1024, 32),
    ('microsoft/deberta-large', 1024, 8),
    ('microsoft/deberta-v3-large', 1024, 8),
    ('allenai/longformer-base-4096', 1024, 32),
    ('google/bigbird-roberta-base', 1024, 32),
    ('google/bigbird-roberta-large', 1024, 8),
]
#------------------------------------------------------------------------------------------------------------------
import os
import numpy as np

path = "/kaggle/input/embeddings-newsgroup-atiml/"
all_train_embeds = []
all_test_embeds = []

for (model, max_length, batch_size) in models:
    name = path + model.replace("/","_") + ".npy"
    if os.path.exists(name):
        train_embed = np.load(name)
        print(train_embed.shape)
        print(f"Loading train embeddings for {name}")
    else:
        print(f"Computing train embeddings for {name}")
        train_embed, test_embed = get_embeddings(model_name=model, max_length=max_length, batch_size=batch_size, compute_train=True)
        np.save(name, train_embed)
    all_train_embeds.append(train_embed)

#del train_embed
#------------------------------------------------------------------------------------------------------------------
all_train_embeds = np.concatenate(all_train_embeds,axis=1)

gc.collect()
print('Our concatenated train embeddings have shape', all_train_embeds.shape )
#------------------------------------------------------------------------------------------------------------------
# # PCA

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



# Step 2: Fit PCA
pca = PCA(n_components=5000)  # We will reduce the data to 2 principal components
pca.fit(all_train_embeds)

pca_data = pca.transform(all_train_embeds)



explained_variance = pca.explained_variance_ratio_


import matplotlib.pyplot as plt
import numpy as np

# Example explained variance values (replace with your actual values)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_[:50])


# Plot the cumulative explained variance as a filled bar graph
plt.figure(figsize=(8, 5))

# Bar graph for cumulative explained variance
bar_plot_explained = plt.bar(range(1, len(explained_variance[:50]) + 1), cumulative_variance, alpha=0.7, label='Cumulative Explained Variance', color='steelblue')

# Fill the inside of the bars with color representing explained variance
#bar_plot = plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Explained Variance', color='b')

# Line graph for cumulative explained variance
plt.plot(range(1, len(explained_variance[:50]) + 1), cumulative_variance, marker='o', linestyle='-', color='r', label='Cumulative Variance', )




# Add labels and title
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance of top 50 Principal Components')
plt.legend()

# Show the plot
plt.show()

# Extract explained variance
explained_variance = pca.explained_variance_ratio_

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, 5001), explained_variance, alpha=0.6, color='b', edgecolor='k')
plt.title('Explained Variance by PCA Components')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()

# ### Naive Bayes Classification
#------------------------------------------------------------------------------------------------------------------
# %% [code] {"execution":{"iopub.status.busy":"2024-06-19T18:27:32.244355Z","iopub.execute_input":"2024-06-19T18:27:32.244708Z","iopub.status.idle":"2024-06-19T18:27:32.902823Z","shell.execute_reply.started":"2024-06-19T18:27:32.244677Z","shell.execute_reply":"2024-06-19T18:27:32.901554Z"}}
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(pca_data[:,:2000])

y_split = pd.DataFrame(train['labels']).astype(int).values

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(scaled_data, y_split.ravel())

y_pred = clf.predict(scaled_data)
accuracy = accuracy_score(y_split, y_pred.reshape(len(y_pred), 1))
print(accuracy)

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=False )

for fold,(train_index, val_index) in enumerate(skf.split(train,train["labels"])):
    
    X_train_fold, X_fold = pca_data[:,:2000][train_index], pca_data[:,:2000][val_index]
    break

y_split_fold = y_split[val_index]

y_split_train = y_split[train_index]

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_fold, y_split_train.ravel())

y_pred = clf.predict(X_fold)
accuracy = accuracy_score(y_split_fold, y_pred.reshape(len(y_pred), 1))
print(accuracy)

# # Cosine Similarity
#------------------------------------------------------------------------------------------------------------------
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import cupy as cp

train_subset = train.iloc[val_index]

from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(stop_words='english',max_features=1024)
train_embed = model.fit_transform(train_subset.full_text.values)
train_embed = cp.array(train_embed.toarray())
norm = cp.sqrt( cp.sum(train_embed*train_embed,axis=1, keepdims=True) )
train_embed = train_embed / norm

cos_similarity = cp.dot(train_embed, train_embed.T)
#top1 = cp.argmax(top1,axis=0)

cos_similarity[np.isclose(cos_similarity, 1.0)] = 0

# Zero out the upper triangular part, excluding the diagonal
cos_similarity[np.triu_indices_from(cos_similarity, k=1)] = 0

# Constraints
#------------------------------------------------------------------------------------------------------------------
indices_ml = np.where(cos_similarity > 0.8)
indices_cl = np.where((cos_similarity != 0) & (cos_similarity < 0.001))

must_link = []
for i in range(len(indices_ml[0])):
    must_link.append(( indices_ml[0][i].tolist(),indices_ml[1][i].tolist() ))

cannot_link = []
for i in range(1000):
    cannot_link.append((indices_cl[0][i].tolist(),indices_cl[1][i].tolist() ))

# # Clustering
#------------------------------------------------------------------------------------------------------------------
from cop_kmeans import cop_kmeans

clusters, centers = cop_kmeans(dataset=X_fold, k=20, ml=must_link, min_size = 100 ,max_size = 200 )

pd.Series(clusters).value_counts()

from sklearn.metrics import silhouette_score

# Calculate Silhouette Score
#------------------------------------------------------------------------------------------------------------------
score = silhouette_score(np.array(clusters).reshape((3770, 1)), y_split_fold)
print(f'Silhouette Score: {score}')


import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 8) 

tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
vis_dims2 = tsne.fit_transform(train_embed)

x = [x for x,y in vis_dims2]
y = [y for x,y in vis_dims2]

palette = sns.color_palette("inferno", 20).as_hex() 

for category, color in enumerate(palette):
    xs = np.array(x)[np.where(y_split.reshape(18846) == category )]
    ys = np.array(y)[np.where(y_split.reshape(18846) == category )]
    plt.scatter(xs, ys, color=color, alpha=0.1)

    avg_x = xs.mean()
    avg_y = ys.mean()
    
    plt.scatter(avg_x, avg_y, marker='x', color=color, s=100)
plt.title("Embeddings visualized using t-SNE")

import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 8) 

tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
vis_dims2 = tsne.fit_transform(X_fold)

x = [x for x,y in vis_dims2]
y = [y for x,y in vis_dims2]

palette = sns.color_palette("deep", 20,)

for category, color in enumerate(palette):
    xs = np.array(x)[np.where(np.array(clusters) == category )]
    ys = np.array(y)[np.where(np.array(clusters) == category )]
    plt.scatter(xs, ys, color=color, alpha=0.1)

    avg_x = xs.mean()
    avg_y = ys.mean()
    
    plt.scatter(avg_x, avg_y, marker='x', color=color, s=100)
plt.title("Embeddings visualized using t-SNE")

indices_center = []
for i in range(20):
    index = np.where(newsgroups.target == i)[0][0]
    indices_center.append(index)
