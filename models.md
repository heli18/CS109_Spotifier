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

### [3. Linear regression](#linear-regression)

#### Linear regression / Linear regression with encoded variables
#### Lasso
#### Gradient Boosted Trees

### [4. Artificial neuronetwork](#artificial-neuronetwork)

### [5. Combined model](#combined-model)
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

#### Linear regression / Linear regression with encoded variables
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
#### Lasso
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

#### Gradient Boosted Trees
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

### Artificial Neuronetwork
### Combined Model