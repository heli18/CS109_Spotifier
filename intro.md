[SPOTIFIER](https://heli18.github.io/CS109_Spotifier/) |
[Introduction and EDA](https://heli18.github.io/CS109_Spotifier/intro) |
[Literature Review and Related Work](https://heli18.github.io/CS109_Spotifier/lit) |
[Models](https://heli18.github.io/CS109_Spotifier/models) |
[Results and Conclusion](https://heli18.github.io/CS109_Spotifier/results) |
[Downloads](https://heli18.github.io/CS109_Spotifier/downloads) 

# SPOTIFIER! : CS109 DATA SCIENCE FINAL PROJECT

## Introduction and EDA

### [1. Introduction and Description of Data](#introduction-and-description-of-data)
#### [a. Description of Raw and Final Data](#description-of-raw-and-final-data)
#### [b. Data Parsing](#data-parsing)
### [2. EDA](#eda)

### Introduction And Description Of Data

Playlist generation is one of the major problems in Music Recommender Systems. The idea is to generate playlists that create the best experience for users based on their preferences, taste, emotion, time of the day, etc. A successful implementation of playlist generation creates more user engagement and extends the amount of time they spent with the Spotify music application.

#### Description of Raw and Final Data

The data being used is from the Million Playlist dataset, which comprises of 1,000,000 playlists generated by Spotify users, with each playlist averaging around 67 songs.

The data is formatted in JSON with playlist properties including: name of the playlist, whether it is collaborative, number or tracks and albums it includes, duration, number of followers it has, the number of edits, and number of artists (see below). The information contained with each track property of each playlist is: artist name, track, name, album, duration of the song.

The Million Playlist dataset had more unnecessary metadata which we cleaned in the data exploration ( track_uri,artist_uri, album_uri). 


#### Data Parsing

Million Playlist dataset format:
```python
"playlists": [
	 {
		 "name": "Throwbacks",
		 "collaborative": "false",
		 "pid": 0, (remove)
		 "modified_at": 1493424000, (remove)
		 "num_tracks": 52,
		 "num_albums": 47,
		 "num_followers": 1,
		 "num_edits": 6,
		 "duration_ms": 11532414, (remove)
		 "num_artists": 37,
		 "durantion_ms_mean": ‘float’ (add)
		 "popular_artist": ‘str’ (add)
		 "popular_album": ‘str’ (add)
		 "tracks": [
			 {
				 "pos": 0, (remove)
				 "artist_name": "Missy Elliott",
				 "track_uri": "spotify:track:0UaMYEvWZi0ZqiDOoHU3YI", (remove)
				 "artist_uri": "spotify:artist:2wIVse2owClT7go1WT98tk", (remove)
				 "track_name": "Lose Control (feat. Ciara & Fat Man Scoop)",
				 "album_uri": "spotify:album:6vV5UrXcfyQD1wu4Qo2I9K", (remove)
				 "duration_ms": 226863,
				 "album_name": "The Cookbook"
		 	},
		 	...
		 ]
 	},
 	...
 ]
```

The data was combined in one dictionary and parsed using the python script below, so that the new data set has the variables we are interested in, while removing data which will not be used in further analysis.

Parsed data format:
```python
"playlists": [
	{
		"name": "Throwbacks",
		"collaborative": "false",
		"popular_album": "Departure - Recharged",
		"num_albums": 47,
		"num_tracks": 52,
		"num_followers": 1,
		"tracks": [
			{
				"artist_name": "Missy Elliott",
				"track_name": "Lose Control (feat. Ciara & Fat Man Scoop)",
				"duration_ms": 226863,
				"album_name": "The Cookbook"
			},
		]
	}
]
```

One issue that we faced was the missing data (null). So, if playlists had <= 30% of track data missing, we dropped the playlists. Since doing so would affect some of the statistics of the playlists (in our case duration of playlists), we decided to use the mean of it. 

### EDA
We first wanted to see if there was a disparity between popular playlists being created by 1 person/company, as opposed to having many collaborators contributing to the popularity of the playlist.

```python
# Get predictors for num_followers
y_num_followers = clean_df_2[['num_followers']]

x_wo_followers = clean_df_2.drop('num_followers', axis=1)
# Check if playlists with collabators have a huge following

plt.violinplot(clean_df_2.loc[clean_df_2.collaborative==0,'num_followers'],positions=[1])
plt.violinplot(clean_df_2.loc[clean_df_2.collaborative==1,'num_followers'],positions=[2])

plt.xlabel("Playlist Collabarations")
plt.xticks([1,2], ["False","True"])
plt.ylabel("Number of Followers")
plt.title('Playlists with Collaborators')
plt.show()
```

![eda_01](https://heli18.github.io/CS109_Spotifier/images/EDA1.png)

As we can tell that playlists with a huge following are actually made by either 1 person or an organization, the popular playlists don't take contributors. 

We also wanted to check if there were any popular playlists with just 1 artist (like playlists dedicated to a specific artist) this could help us notice if popular artists can have a strong enough following to not need other artists music to increase their following.

```python
try:
    plt.violinplot(clean_df_2.loc[clean_df_2.num_artists==1,'num_followers'],positions=[1])
except ValueError:  #raised if `x` is empty.
    pass

plt.violinplot(clean_df_2.loc[clean_df_2.num_artists > 1,'num_followers'],positions=[2])

plt.xlabel("Number of Artists Per Playlist")
plt.title("Playlists containing just 1 artist")
plt.xticks([1,2], ["1",">1"])
plt.ylabel("Followers")
plt.show()
```

![eda_02](https://heli18.github.io/CS109_Spotifier/images/EDA2.png)

We can notice that even popular artists don't have enough of a strong following to pull in a large amount of followers or that not many people create a playlist based solely on one artist.

Now that we have curated our data to be more manageable, we decided to take a look at possible similarities with the data we currently have.

```python
# Check features
cols_wanted = [
    'collaborative',
    'name',
    'num_artists', 
    'num_edits', 
    'num_followers', 
    'num_tracks', 
    'duration_ms_mean', 
    'duration_min_mean',
    'popular_artist' # doesnt account for categorical variable so this actually never gets displayed
]

scatter_matrix(clean_df_2[cols_wanted], alpha=0.5, figsize=(25,20));
```

![eda_03](https://heli18.github.io/CS109_Spotifier/images/EDA3.png)

We decided to take a look at the most popular artists and see how much of the dataset it actually takes up.

```python
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
plt.title("Top 5 artist Occurrences in Playlists Sample")
plt.ylabel("Occurances in Playlists")
plt.xlabel("Top 5 Artists")

plt.bar(art_sample_5.keys(), art_sample_5.values)
plt.show()
```

```
Top X artists exist in :
786 rows
```

![eda_04](https://heli18.github.io/CS109_Spotifier/images/EDA4.png)


```python
untouched_df_clean = clean_df_2.copy()
```

Noticing that the popular artists make up a decent portion of our dataset we decided to explore possibilities, by isolating only the top 5 popular artists and see if we can notice some trends


```python
# Dataframe with only popular X artists so we can compare categorical variables
pop_df = clean_df_2.loc[clean_df_2['popular_artist'].isin(art_sample_5.keys())]
pop_df.head()
```

![eda_05](https://heli18.github.io/CS109_Spotifier/images/EDA5.png)

```python
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
pop_df.head()
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

![eda_06](https://heli18.github.io/CS109_Spotifier/images/EDA6.png)

In this example we can see that even though that there is top 5 artists, the top 5 artists have up to 503 top songs. Perhaps if we separate by top songs rather than artists we can see some trends

```python
# Run a scatter matrix on the top 5 artist Dataframe to see any interesting occurences 
cols_wanted2 = [
    'collaborative',
    'name',
    'num_artists', 
    'num_edits', 
    'num_followers', 
    'num_tracks', 
    'duration_ms_mean', 
    'duration_min_mean',
    'popular_artist'
]

scatter_matrix(pop_df[cols_wanted2], alpha=0.5, figsize=(25,20));
```

![eda_07](https://heli18.github.io/CS109_Spotifier/images/EDA7.png)

Since there is not anything necessarily out of the ordinary in the scatter matrix above as the spread is too large, we begin to go back to the drawing board and look at classifying our data given the predictors we have. 

```python
print(untouched_df_clean.shape)
untouched_df_clean.head()
```

![eda_08](https://heli18.github.io/CS109_Spotifier/images/EDA8.png)

We first decided to use `num_followers` as our response variable with our predictors only being numerical variables and removing the other cateogorical variables.
