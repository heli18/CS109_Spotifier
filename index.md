[SPOTIFIER](https://heli18.github.io/CS109_Spotifier/) |
[Introduction and EDA](https://heli18.github.io/CS109_Spotifier/intro) |
[Literature Review and Related Work](https://heli18.github.io/CS109_Spotifier/lit) |
[Models](https://heli18.github.io/CS109_Spotifier/models) |
[Results and Conclusion](https://heli18.github.io/CS109_Spotifier/results) |
[Downloads](https://heli18.github.io/CS109_Spotifier/downloads) 

# SPOTIFIER! : CS109 DATA SCIENCE FINAL PROJECT

## Group #3: Heidi Li, Enisa Serhati, Yilun Chen, David Nunez

### Background

Playlist generation is one of the major problems in Music Recommender Systems. The idea is to generate playlists that create the best experience for users based on their preferences, taste, emotion, time of the day, etc. A successful implementation of playlist generation creates more user engagement and extends the amount of time they spent with the Spotify music application.

### Research Goal

This project primarily addresses the question of what the best way to predict user’s track preference is. The primary dataset being used is Million Playlist Dataset. After the visualization and analysis above, we decided to base our prediction on user behavior. Therefore, the main question in this project that we intend to answer is: what predictors in the playlist can well predict the user preference?

### Research Strategy

We focus our research on the Million Playlist Dataset. The strategy we adopt is based on predicting one playlist by another playlist. We split the whole playlist dataset into two parts: one as our playlist bank where contains a large amount of playlists; the other is called training playlist. We identify a user and his/ her music preference or behaviour based on his/ her playlist. And we hide some of the tracks in the training playlist as the ones we need to predict.

In order to predict the tracks that the users would probably like by looking at the tracks in the playlists that are similar to the user’s existing playlist, we need to define effective predictors in the playlist and compute similarity score between a pair of playlist. We would like to explore dimensional reduction method, such as PCA or classification techniques to determine efficient predictors.

However, one challenge for only using the million playlist dataset is the “cold start” problem. Since we have no prior playlist to define a user preference. Hence, we would like to address this problem by recommending popular and hot tracks. These tracks can be predicted by looking into the playlists with a huge amount of followers.