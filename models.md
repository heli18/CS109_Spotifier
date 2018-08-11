[SPOTIFIER](https://heli18.github.io/CS109_Spotifier/) |
[Introduction and EDA](https://heli18.github.io/CS109_Spotifier/intro) |
[Literature Review and Related Work](https://heli18.github.io/CS109_Spotifier/lit) |
[Models](https://heli18.github.io/CS109_Spotifier/models) |
[Results and Conclusion](https://heli18.github.io/CS109_Spotifier/results) |
[Downloads](https://heli18.github.io/CS109_Spotifier/downloads) 

# SPOTIFIER! : CS109 DATA SCIENCE FINAL PROJECT

## Models

We used various models to explore the best way to identify the similarity between the tracks and predict similar songs, to be able to generate a final recommendation playlist. In modeling we included a sample dataset of 10,000 playlists from the 1 million playlists dataset. We split the sample dataset into a Playlist Bank and a Training set. The Playlist Bank contains 90% of the data (9,000) playlists and the Training Set contains 10% of the data (1,000) playlists.

More information related to model training and performance can be found on the [Conclusions and Results](https://heli18.github.io/CS109_Spotifier/results) page.

### [1. Reading and Cleaning Data](#reading-and-cleaning-data)

### [2. Baseline Model](#baseline-model)

### [3. Linear Regression](#linear-regression)

### [4. Lasso](#lasso)

### [5. Gradient Boosted Trees](#gradient-boosted-trees)

### [6. Artificial Neural Network](#artificial-neural-network)

### [7. Combined model](#combined-model)
Combine baseline and advanced model together to generate the final recommendation list.

### Reading and Cleaning Data
``` python
import json
import os
import config
import datetime

LOGFILE = config.log_file

def parse_durations(elements):
	# parse mean duration (total duration / total tracks)
	elements["duration_ms_mean"] = (elements["duration_ms"]/elements["num_tracks"])
	return (elements)

def delete_properties(properties, elements):
	for prop in properties:
		elements.pop(prop, None)
	return elements

def get_popular(items):
	return str(max(set(items), key=items.count).encode('utf-8'))

def parse_files(input_file, output_file):
	properties_to_remove = ["pid", "modified_at", "duration_ms"]
	tracks_properties_to_remove = ["pos", "track_uri", "artist_uri", "album_uri"]

	# read file
	with open(input_file) as f:
	    data = json.load(f)

	# process file
	for element in data["playlists"]:
		artists = []
		albums = []

		# parse durations
		element = parse_durations(element)

		# get popular artist and album
		for track_prop in element["tracks"]:
			artists.append(track_prop["artist_name"])
			albums.append(track_prop["album_name"])
		element["popular_artist"] = (get_popular(artists))
		element["popular_album"] = (get_popular(albums))

		# delete playlist properties
		element = delete_properties(properties_to_remove, element)

		# delete tracks properties
		for track_prop in element["tracks"]:
			element = delete_properties(tracks_properties_to_remove, track_prop)

	# write file
	with open(output_file, 'w') as f:
			json.dump(data, f, indent=4)

with open(LOGFILE, 'w') as log:
	s = str(datetime.datetime.now()) + "\n"

	for items in config.file_dict:
		for file in config.file_dict[items]:
			s+=("parsing input: \"" + file + "\" output: \"" + config.output_dir+"/"+file + "\"\n")
			parse_files(config.files_dir+"/"+file, config.output_dir+"/"+file)

	s += "completed at: " + str(datetime.datetime.now()) + "\n"
	log.write(s)

```

### Baseline Model
``` python
#split the data into playlist bank (90%) and training set (10%)
playlist_bank = {}
for user in range(n_playlist*900):
    playlist_bank[user] = list(data_full[data_full['user']==user]['track'])

playlist_data = {}
for user in range(n_playlist*900, n_playlist*1000):
    playlist_data[user] = list(data_full[data_full['user']==user]['track'])

#define the similarity function
#input: two list with items
#output: the similairty score of two lists
def cal_similarity(list_one, list_two):
    no_overlap = list(Counter(list_one)-Counter(list_two))
    similarity = ((len(list_one)-len(no_overlap))*2)/(len(list_one)+len(list_two))
    return similarity

# Create the function to hide the first track in each playlist
#input: the dictionary of playlists
#output: a list of the hidden track and a dictionary of users and the unhidden tracks
def hide_track(playlists):
    track_hide = []
    track_train = {}
    for key in playlists:
        track_hide.append(playlists[key][0])
        track_train[key] = playlists[key][1:]
    return track_hide, track_train

# Create the function of getting the recommendation list
#input: the playlists that is to be predicted
#       the playlist bank
#       the number of similar playlist that need to be generated
#output: the list of recommendation tracks

def get_recommendation_list(track_train, playlist_bank, no_similar_playlist):
    recommendation_track = []
    time_train = 0
    for ut in track_train:
        sim_score = np.zeros(len(playlist_bank))
        time_bank = 0
        for ub in playlist_bank:
            sim_score[time_bank] = cal_similarity(track_train[ut], playlist_bank[ub])
            time_bank += 1
        similarity_df = pd.DataFrame(sim_score, index=range(len(sim_score)))
        similarity_df = similarity_df.sort_values(by=0, ascending=False)
        index = list(similarity_df.index[[x for x in range(no_similar_playlist)]])
        sample_names = []
        for i in index:
            sample_names.append(playlist_bank[i])
        recommendation_track.append([track for playlist in sample_names for track in playlist])
        time_train += 1
    return recommendation_track    
#Get the list of hidden track and the dictionary of training tracks
track_hide, track_train_base = hide_track(playlist_data)
#Get the top 5 similar playlists as our base of recommendation list
#Randomly select 100 of them as our final recommendation
recommendation_list = get_recommendation_list(track_train_base, playlist_bank, 5)
no_recommendation = 100
#Calculate the score of the prediction
scores = np.zeros(len(track_train_base))
for n in range(len(track_hide)):
    not_overlap = list(Counter(recommendation_list[n])-Counter(track_train_base[n+900*m_adjust]))
    np.random.shuffle(not_overlap)
    recommendation = np.array(not_overlap)[:no_recommendation]
    if track_hide[n] in recommendation:
        scores[n] = 1
score = np.mean(scores)    
print('The score is {0}'.format(score))
```

```
The score is 0.166
```
### Linear Regression

``` python
# # Alloc all playlists as array of dictionaries
all_playlists = []
for i,j in enumerate(complete_data):
    for k,l in enumerate(j['playlists']):
        all_playlists.append(l) 
print("Total number of playlists: ", len(all_playlists))
unclean_df = pd.DataFrame(all_playlists)

## Count all top songs in tracks array
unclean_copy = unclean_df.copy()
def get_popular(items):
    return str(max(set(items), key=items.count))

t_song = []
t_dict = {}

for element in unclean_copy['tracks']:
    top_song_artist = []
    
    for track in element:
        key = track['track_name']
    
        if key in t_dict:
            t_dict[key] += 1
        else:
            t_dict[key] = 1

ordered_songs = OrderedDict(sorted(t_dict.items(), key=itemgetter(1), reverse=True))
print("We are observing over {} different songs".format(len(ordered_songs)))
```

```
We are observing over 132920 different songs
```

```python
clean_df = unclean_copy.drop(['description', 'tracks'], axis=1 )
# Get a column of duration_min_mean based on millisecond column
dur_ms_mean = clean_df[['duration_ms_mean']]

min_arr = []
for k,v in dur_ms_mean.items():
    min_arr.append(v/60000)
    
# Append new Minute data column
tz = np.asarray(min_arr)
clean_df['duration_min_mean'] = tz[0]
d = {'false': 0, 'true': 1}
tmp = clean_df['collaborative'].map(d)

clean_df['collaborative'] = tmp.values

# Copy prev dataframe
clean_df_2 = clean_df.copy()

# Get predictors for num_followers
y_num_followers = clean_df_2[['num_followers']]

x_wo_followers = clean_df_2.drop('num_followers', axis=1)

# most popular artists among the playlists
pop_art = clean_df_2['popular_artist'].value_counts()
print("Unique Artists in Sample: ",len(pop_art))

art_sample_5 = pop_art[:5]
print("Top 5 artists in Sample make up: {} of the sample dataset. \nThe top 5 artists are: \n{} ".format(sum(art_sample_5),pop_art[:5]) )

print("")
print("Out of: {} playlists".format(sum(clean_df_2['popular_artist'].value_counts())) )

print("Top 5 artists make up: {}% of dataset ".format(sum(art_sample_5)/sum(clean_df_2['popular_artist'].value_counts()) * 100))
```

```
Unique Artists in Sample:  2874
Top 5 artists in Sample make up: 786 of the sample dataset. 
The top 5 artists are: 
Drake               361
The Chainsmokers    128
Beyoncé             101
Rihanna             100
Ed Sheeran           96
Name: popular_artist, dtype: int64 

Out of: 10000 playlists
Top 5 artists make up: 7.86% of dataset 
```

```python
art_sample_5.keys
art_sample_5.values

print("Top X artists exist in :")
print(len(clean_df_2[clean_df_2['popular_artist'].isin(art_sample_5.keys())]), "rows")
```

```
Top X artists exist in :
786 rows
```

```python
untouched_df_clean = clean_df_2.copy()
# Dataframe with only popular X artists so we can compare categorical variables
pop_df = clean_df_2.loc[clean_df_2['popular_artist'].isin(art_sample_5.keys())]
# Turn the categorical artist names to numerical values so we can better compare against
unique_artists_pop = set(pop_df['popular_artist'].values)
print(unique_artists_pop)
art_int_cols = {}

counter = 0
for i in unique_artists_pop:
    art_int_cols[i] = counter
    counter+=1

print("Top 5 artists by key:",art_int_cols)

tmp3 = pop_df['popular_artist'].map(art_int_cols)
print(tmp3[:5])

pop_df['popular_artist'] = tmp3

print("Shape of our dataset: ",pop_df.shape)
```

```
{'Beyoncé', 'Rihanna', 'Ed Sheeran', 'The Chainsmokers', 'Drake'}
Top 5 artists by key: {'Beyoncé': 0, 'Ed Sheeran': 2, 'Rihanna': 1, 'The Chainsmokers': 3, 'Drake': 4}
34    3
35    3
38    1
40    0
60    1
Name: popular_artist, dtype: int64
Shape of our dataset:  (786, 11)
```
```python
# Clean up categorical variables 
clean_categorical_df = untouched_df_clean.drop(['name', 'popular_album', 'popular_artist'], axis=1 )

# Split into test datasets
X_train, X_test, y_train, y_test = train_test_split(clean_categorical_df.drop('num_followers', axis=1),clean_categorical_df['num_followers'], test_size=0.33, random_state=42 )

# Fit a Multilinear Regression Model
linreg = LinearRegression()
linreg.fit(X_train, y_train)
linreg.score(X_test, y_test)
#We can see the 3rd columns negative coefficients is far different than the rest which seem pretty calm
linreg.coef_
```

```
-0.0038291298743347024
array([-2.48097443e+00, -4.85389761e-05, -2.75847547e-02,  1.47803917e-01,
       -1.44966348e-01, -2.22800770e-02, -8.08982975e-10])
```

```python
# We check the first 10 items in our predictions against what it should result to in the y_test
print("Predictions (first 10):", linreg.predict(X_test)[:10])
print("")
print("True values (first 10):\n{}".format(y_test[:10]))
```

```
Predictions (first 10): [ 4.16747251  4.29853431  9.71456744  6.70981929  1.10323296  3.78311267
  6.11011493  5.04846683 -3.86752462  6.47423513]

True values (first 10):
6252    1
4684    1
1731    1
4742    2
4521    1
6340    1
576     1
5202    1
6363    1
439     2
Name: num_followers, dtype: int64
```

```python
print("Display the outliers in our Test Set to see what may be ruining our model (Lowest to Highest):")
print(np.argsort(y_test)[::][:5])
```

```
Display the outliers in our Test Set to see what may be ruining our model (Lowest to Highest):
6252       0
4684    2038
1731    2040
4742    2043
4521    2044
Name: num_followers, dtype: int64
```

```python
print("Display the outliers in our Test Set to see what may be ruining our model (Highest to Lowest):")
print(np.argsort(y_test)[::-1][:5])
```

```
Display the outliers in our Test Set to see what may be ruining our model (Highest to Lowest):
9401    2490
8781     117
6094    1253
9754      86
1744    1196
Name: num_followers, dtype: int64
```

```python
# Create a new dataframe to better see how the outliers differ compared to the other predictors.
temp = X_test.copy()
temp['num_followers'] = y_test

temp.sort_values(['num_followers'], ascending=False, inplace=True)

# Create clean dataframes
df_newclean = untouched_df_clean.copy()

temp2 = df_newclean.copy()
temp2.sort_values(['num_followers'], ascending=False, inplace=True)

# hot encode popular artist categorical variable
df_newclean = untouched_df_clean.copy()
dummy_art = pd.get_dummies(df_newclean['popular_artist'])
print("Shape of hot encoded Categorical variable Popular Artist",dummy_art.shape)

# merge
df_newclean.shape
merge_df = pd.concat([df_newclean, dummy_art], axis=1)
```

```
Shape of hot encoded Categorical variable Popular Artist (10000, 2874)
```

```python
X_train, X_test, y_train, y_test = train_test_split(merge_df.drop(['num_followers','name', 'popular_album', 'popular_artist'], axis=1),merge_df['num_followers'], test_size=0.33, random_state=42 )

lr = LinearRegression()

lr.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(merge_df.drop(['num_followers','name', 'popular_album', 'popular_artist'], axis=1),merge_df['num_followers'], test_size=0.33, random_state=42 )

lr = LinearRegression()

lr.fit(X_train, y_train)
```

```
Accuracy Score (Test): -50960798734952.555

Coefficients: [-9.41829635e-02  5.25917963e+02 -1.79569798e-01 ... -4.49115342e+07
  0.00000000e+00 -4.49115063e+07]

Accuracy Score(Train): 0.037381214596123535
```
### Lasso
```python
# use LASSO
las = Lasso(alpha=0.5)
las.fit(X_train, y_train)
print(
    "Accuracy Score (Test): {}\n\nCoefficients: {}\n\nAccuracy Score(Train): {}\n"
    .format(
    las.score(X_test, y_test),
    las.coef_,
    las.score(X_train, y_train))
)
```

```
Accuracy Score (Test): -0.43577876866358367

Coefficients: [-0.00000000e+00 -3.10942535e-05 -5.78218784e-02 ... -0.00000000e+00
  0.00000000e+00  0.00000000e+00]

Accuracy Score(Train): 0.03539105570048673
```

### Gradient Boosted Trees
```python
xgbr = xgb.XGBRegressor(max_depth=10, 
                        learning_rate=0.1, 
                        n_estimators=60, 
                        silent=True, 
                        min_child_weight=1,
                        objective='reg:linear')
xgbr.fit(X_train, y_train, eval_metric='rmse', verbose = False, eval_set = [(X_train,y_train),(X_test, y_test)],early_stopping_rounds=10)
xgbr.score(X_test, y_test)
xgbr.score(X_train, y_train)

```
```
-11.147813673648637
0.6763274370298381
```

### Artificial Neural Network
```python
def get_matrix(data, no_interaction):
    num_user = len(data['user'].unique())
    num_track = len(data['track'].unique())
    data_matrix = sp.dok_matrix((num_user, num_track), dtype=np.float32)
    for u, t in zip(data['user'], data['track']):
        data_matrix[u, t] = 1.0
    return num_user, num_track, data_matrix
#get the random features
#input: the data matrix
#output: the desighed user, track input and response variable
def get_features(data_matrix):
    num_user, num_track = data_matrix.shape
    user_data, track_data, y_data = [],[],[]
    
    for (u, i) in data_matrix.keys():
        user_data.append(u)
        track_data.append(i)
        y_data.append(1)
        
        for t in range(no_interaction):
            j = np.random.randint(num_track)
            while (u,j) in data_matrix:
                j = np.random.randint(num_user)
            user_data.append(u)
            track_data.append(j)
            y_data.append(0)
                
    return user_data, track_data, y_data
#build the model
def build_model(num_user, num_track, latent_dim):
    user_train = Input(shape=(1,), dtype='int32', name = 'user_train')
    track_train = Input(shape=(1,), dtype='int32', name = 'track_train')

    MF_Embedding_user = Embedding(input_dim = num_user, output_dim = latent_dim, embeddings_initializer='uniform', embeddings_regularizer = l2([0,0][0]), input_length=1)
    MF_Embedding_track = Embedding(input_dim = num_track, output_dim = latent_dim, embeddings_initializer='uniform', embeddings_regularizer = l2([0,0][0]), input_length=1)   
    
    user_latent = Flatten()(MF_Embedding_user(user_train))
    track_latent = Flatten()(MF_Embedding_track(track_train))

    predictor = add([user_latent, track_latent])
    prediction = Dense(1, activation="sigmoid", name="prediction", kernel_initializer="lecun_uniform")(predictor)
    NNmodel = Model(inputs=[user_train, track_train], outputs=prediction)
    return NNmodel
#get matrix
no_interaction = 1
num_user, num_track, matrix = get_matrix(data_adjust, no_interaction)
user, track, y = get_features(matrix)
data_nn = pd.DataFrame(user, columns=['user'])
data_nn['track'] = track
data_nn['y'] = y

#get train and test data
data_copy = data_nn.copy()
data_test = pd.DataFrame(columns=['user', 'track', 'y'])
for n in range(n_adjust*900, n_adjust*1000):
    data_hide = data_copy[data_copy.user == n].iloc[0:2]
    data_test = data_test.append(data_hide)
    
data_train = data_copy.drop(data_test.index)

user_train = data_train.user
track_train = data_train.track
y_train = data_train.y

user_test = data_test.user
track_test = data_test.track
y_test = data_test.y

#define parameter range
latent_dims = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
epochss = [3, 20]
```

#### Latent Dimension Selection
```python
learning_rate = learning_rates[1]
epochs = epochss[0]
latent_dim_list, train_score_list, test_score_list = [], [], []
for latent_dim in latent_dims:
    NNmodel = build_model(num_user, num_track, latent_dim)
    NNmodel.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    NNmodel.fit([np.array(user_train), np.array(track_train)], np.array(y_train), epochs=epochs, batch_size=200, validation_split = .2)
    NN_train_score = NNmodel.evaluate([np.array(user_train), np.array(track_train)], np.array(y_train))[1]
    NN_test_score = NNmodel.evaluate([np.array(user_test), np.array(track_test)], np.array(y_test))[1]
    latent_dim_list.append(latent_dim)
    train_score_list.append(NN_train_score)
    test_score_list.append(NN_test_score)
```
```
Train on 104548 samples, validate on 26138 samples
Epoch 1/3
104548/104548 [==============================] - 1s 12us/step - loss: 0.6927 - acc: 0.5136 - val_loss: 0.6915 - val_acc: 0.5341
Epoch 2/3
104548/104548 [==============================] - 1s 5us/step - loss: 0.6877 - acc: 0.5836 - val_loss: 0.6880 - val_acc: 0.5595
Epoch 3/3
104548/104548 [==============================] - 1s 5us/step - loss: 0.6784 - acc: 0.6439 - val_loss: 0.6829 - val_acc: 0.5735
130686/130686 [==============================] - 2s 12us/step
200/200 [==============================] - 0s 87us/step
Train on 104548 samples, validate on 26138 samples
Epoch 1/3
104548/104548 [==============================] - 1s 9us/step - loss: 0.6922 - acc: 0.5156 - val_loss: 0.6893 - val_acc: 0.5446
Epoch 2/3
104548/104548 [==============================] - 1s 7us/step - loss: 0.6810 - acc: 0.6073 - val_loss: 0.6825 - val_acc: 0.5719
Epoch 3/3
104548/104548 [==============================] - 1s 7us/step - loss: 0.6627 - acc: 0.6714 - val_loss: 0.6746 - val_acc: 0.5872
130686/130686 [==============================] - 1s 11us/step
200/200 [==============================] - 0s 22us/step
...
...
Train on 104548 samples, validate on 26138 samples
Epoch 1/3
104548/104548 [==============================] - 21s 198us/step - loss: 0.6870 - acc: 0.5355 - val_loss: 0.6713 - val_acc: 0.5847
Epoch 2/3
104548/104548 [==============================] - 19s 183us/step - loss: 0.6220 - acc: 0.6738 - val_loss: 0.6780 - val_acc: 0.5909
Epoch 3/3
104548/104548 [==============================] - 21s 199us/step - loss: 0.5653 - acc: 0.7116 - val_loss: 0.7372 - val_acc: 0.5901
130686/130686 [==============================] - 2s 16us/step
200/200 [==============================] - 0s 55us/step
Train on 104548 samples, validate on 26138 samples
Epoch 1/3
104548/104548 [==============================] - 24s 232us/step - loss: 0.6867 - acc: 0.5357 - val_loss: 0.6690 - val_acc: 0.5925
Epoch 2/3
104548/104548 [==============================] - 23s 222us/step - loss: 0.6184 - acc: 0.6742 - val_loss: 0.6813 - val_acc: 0.5903
Epoch 3/3
104548/104548 [==============================] - 23s 221us/step - loss: 0.5632 - acc: 0.7120 - val_loss: 0.7406 - val_acc: 0.5883
130686/130686 [==============================] - 2s 18us/step
200/200 [==============================] - 0s 27us/step
```
#### Learning Rate Selection
```python
latent_dim = 10
epochs = epochss[0]
learning_rate_list, train_score_list, test_score_list = [], [], []
for learning_rate in learning_rates:
    NNmodel = build_model(num_user, num_track, latent_dim)
    NNmodel.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    NNmodel.fit([np.array(user_train), np.array(track_train)], np.array(y_train), epochs=epochs, batch_size=200, validation_split = .2)
    NN_train_score = NNmodel.evaluate([np.array(user_train), np.array(track_train)], np.array(y_train))[1]
    NN_test_score = NNmodel.evaluate([np.array(user_test), np.array(track_test)], np.array(y_test))[1]
    learning_rate_list.append(learning_rate)
    train_score_list.append(NN_train_score)
    test_score_list.append(NN_test_score)
```

```
Train on 104548 samples, validate on 26138 samples
Epoch 1/3
104548/104548 [==============================] - 3s 27us/step - loss: 0.6930 - acc: 0.5051 - val_loss: 0.6923 - val_acc: 0.5184
Epoch 2/3
104548/104548 [==============================] - 2s 20us/step - loss: 0.6906 - acc: 0.5512 - val_loss: 0.6910 - val_acc: 0.5389
Epoch 3/3
104548/104548 [==============================] - 2s 19us/step - loss: 0.6873 - acc: 0.5958 - val_loss: 0.6892 - val_acc: 0.5540
130686/130686 [==============================] - 2s 17us/step
200/200 [==============================] - 0s 24us/step
Train on 104548 samples, validate on 26138 samples
Epoch 1/3
104548/104548 [==============================] - 3s 30us/step - loss: 0.6908 - acc: 0.5279 - val_loss: 0.6855 - val_acc: 0.5675
Epoch 2/3
104548/104548 [==============================] - 2s 22us/step - loss: 0.6652 - acc: 0.6525 - val_loss: 0.6684 - val_acc: 0.5883
Epoch 3/3
104548/104548 [==============================] - 2s 22us/step - loss: 0.6178 - acc: 0.7065 - val_loss: 0.6638 - val_acc: 0.5914
130686/130686 [==============================] - 2s 17us/step
200/200 [==============================] - 0s 26us/step
...
...
Train on 104548 samples, validate on 26138 samples
Epoch 1/3
104548/104548 [==============================] - 3s 30us/step - loss: 0.6989 - acc: 0.5325 - val_loss: 0.6749 - val_acc: 0.5800
Epoch 2/3
104548/104548 [==============================] - 2s 19us/step - loss: 0.6343 - acc: 0.6476 - val_loss: 0.7392 - val_acc: 0.5808
Epoch 3/3
104548/104548 [==============================] - 2s 21us/step - loss: 0.5939 - acc: 0.6863 - val_loss: 0.7592 - val_acc: 0.5786
130686/130686 [==============================] - 2s 16us/step
200/200 [==============================] - 0s 31us/step
Train on 104548 samples, validate on 26138 samples
Epoch 1/3
104548/104548 [==============================] - 3s 27us/step - loss: 1.0304 - acc: 0.5049 - val_loss: 1.2969 - val_acc: 0.5254
Epoch 2/3
104548/104548 [==============================] - 2s 20us/step - loss: 1.1891 - acc: 0.5376 - val_loss: 1.1681 - val_acc: 0.5083
Epoch 3/3
104548/104548 [==============================] - 2s 19us/step - loss: 1.2318 - acc: 0.5767 - val_loss: 1.5914 - val_acc: 0.5637
130686/130686 [==============================] - 2s 17us/step
200/200 [==============================] - 0s 50us/step
```
#### Epochs Selection
```python
latent_dim = 10
learning_rate = 0.001
epochs = 20
NNmodel = build_model(num_user, num_track, latent_dim)
NNmodel.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
NNmodel.fit([np.array(user_train), np.array(track_train)], np.array(y_train), epochs=epochs, batch_size=200, validation_split = .2)
```

```
Train on 104548 samples, validate on 26138 samples
Epoch 1/20
104548/104548 [==============================] - 4s 40us/step - loss: 0.6883 - acc: 0.5331 - val_loss: 0.6731 - val_acc: 0.5827
Epoch 2/20
104548/104548 [==============================] - 2s 23us/step - loss: 0.6277 - acc: 0.6727 - val_loss: 0.6668 - val_acc: 0.5945
...
...
Epoch 19/20
104548/104548 [==============================] - 2s 20us/step - loss: 0.4960 - acc: 0.7457 - val_loss: 1.1711 - val_acc: 0.5887
Epoch 20/20
104548/104548 [==============================] - 2s 21us/step - loss: 0.4952 - acc: 0.7469 - val_loss: 1.1909 - val_acc: 0.5884
```
#### Create the Final Neural Network Model
```python
#Use the optimal parameter and train the final model
latent_dim = 10
learning_rate = 0.001
epochs = 5
#get matrix of the full dataset
no_interaction = 1
num_user, num_track, matrix = get_matrix(data_full, no_interaction)
user, track, y = get_features(matrix)
data_nn = pd.DataFrame(user, columns=['user'])
data_nn['track'] = track
data_nn['y'] = y

#get train and test data
data_copy = data_nn.copy()
data_test = pd.DataFrame(columns=['user', 'track', 'y'])
for n in range(n_playlist*900, n_playlist*1000):
    data_hide = data_copy[data_copy.user == n].iloc[0:2]
    data_test = data_test.append(data_hide)
    
data_train = data_copy.drop(data_test.index)

user_train = data_train.user
track_train = data_train.track
y_train = data_train.y

user_test = data_test.user
track_test = data_test.track
y_test = data_test.y
#Build the final model
NNmodel = build_model(num_user, num_track, latent_dim)
NNmodel.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
print(NNmodel.summary())
```

```
________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
user_train (InputLayer)         (None, 1)            0                                            
__________________________________________________________________________________________________
track_train (InputLayer)        (None, 1)            0                                            
__________________________________________________________________________________________________
embedding_53 (Embedding)        (None, 1, 10)        100000      user_train[0][0]                 
__________________________________________________________________________________________________
embedding_54 (Embedding)        (None, 1, 10)        1329200     track_train[0][0]                
__________________________________________________________________________________________________
flatten_53 (Flatten)            (None, 10)           0           embedding_53[0][0]               
__________________________________________________________________________________________________
flatten_54 (Flatten)            (None, 10)           0           embedding_54[0][0]               
__________________________________________________________________________________________________
add_27 (Add)                    (None, 10)           0           flatten_53[0][0]                 
                                                                 flatten_54[0][0]                 
__________________________________________________________________________________________________
prediction (Dense)              (None, 1)            11          add_27[0][0]                     
==================================================================================================
Total params: 1,429,211
Trainable params: 1,429,211
Non-trainable params: 0
__________________________________________________________________________________________________
None

```

```python
#Fit the model
NNmodel.fit([np.array(user_train), np.array(track_train)], np.array(y_train), epochs=epochs, batch_size=200, validation_split = .2)
NN_train_score = NNmodel.evaluate([np.array(user_train), np.array(track_train)], np.array(y_train))[1]
NN_test_score = NNmodel.evaluate([np.array(user_test), np.array(track_test)], np.array(y_test))[1]
```

```
Train on 1042625 samples, validate on 260657 samples
Epoch 1/5
1042625/1042625 [==============================] - 90s 86us/step - loss: 0.5454 - acc: 0.7226 - val_loss: 0.4986 - val_acc: 0.7673
Epoch 2/5
1042625/1042625 [==============================] - 89s 85us/step - loss: 0.4562 - acc: 0.7914 - val_loss: 0.5134 - val_acc: 0.7667
Epoch 3/5
1042625/1042625 [==============================] - 92s 88us/step - loss: 0.4398 - acc: 0.8005 - val_loss: 0.5287 - val_acc: 0.7653
Epoch 4/5
1042625/1042625 [==============================] - 88s 85us/step - loss: 0.4311 - acc: 0.8047 - val_loss: 0.5379 - val_acc: 0.7649
Epoch 5/5
1042625/1042625 [==============================] - 93s 89us/step - loss: 0.4254 - acc: 0.8073 - val_loss: 0.5477 - val_acc: 0.7639
1303282/1303282 [==============================] - 27s 20us/step
2000/2000 [==============================] - 0s 21us/step
```

```python
print('The NN train score is {}'.format(round(NN_train_score, 2)))
print('The NN test score is {}'.format(round(NN_test_score, 2)))
```

```
he NN train score is 0.81
The NN test score is 0.75
```

```python
min_train_score = 1 - np.mean(y_train)
min_test_score = 1 - np.mean(y_test)
print('The minimum train score is {}'.format(round(min_train_score, 2)))
print('The minimum test score is {}'.format(round(min_test_score, 2)))

```

```
The minimum train score is 0.5
The minimum test score is 0.5
```

### Combined Model

```python
recom_list_combine = recommendation_list.copy()
recommendation_scores = np.zeros(len(track_hide))
for x in range(len(track_hide)):
    track_full_list = list(set(recom_list_combine[x]))
    existing_track = data_train[data_train['user'] == x].track
    hidden_track = list(track_test)[x*2]
    predict_track = list(Counter(track_full_list)-Counter(existing_track))
    predict_user = np.zeros(len(predict_track))+x
    
    s = NNmodel.predict([np.array(x).reshape(-1,1), np.array(hidden_track).reshape(-1,1)])
    #print('score is {}'.format(s))
    
    predictions = np.transpose(NNmodel.predict([predict_user, np.array(predict_track)]))[0]
    index = np.argsort(predictions)[-no_recommendation:]

    recommendation_list_ann = np.array(predict_track)[index]
    
    if hidden_track in recommendation_list_ann:
        recommendation_scores[x] = 1
        
total_score = np.average(recommendation_scores)
print('For {} percent of time, we successfully include the hidden track'.format(round(total_score*100)))
```

```
For 26.0 percent of time, we successfully include the hidden track
```