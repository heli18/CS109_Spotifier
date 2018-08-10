[SPOTIFIER](https://heli18.github.io/CS109_Spotifier/) |
[Introduction and EDA](https://heli18.github.io/CS109_Spotifier/intro) |
[Literature Review and Related Work](https://heli18.github.io/CS109_Spotifier/lit) |
[Models](https://heli18.github.io/CS109_Spotifier/models) |
[Results and Conclusion](https://heli18.github.io/CS109_Spotifier/results) |
[Downloads](https://heli18.github.io/CS109_Spotifier/downloads) 

# SPOTIFIER! : CS109 DATA SCIENCE FINAL PROJECT

## Literature Review and Related Work

### Literature Review

Currently Spotify generates playlists by relying on collaborative ‘filtering’, a method which finds similarities in users instead of songs. So to recommend songs to User X, Spotify finds other users with similar listening record and recommends those songs from other users to User X. As for the issue of Cold Start Problem, Spotify uses a neural network analyses of sound files and determines songs that have similar patterns. Additionally, Spotify uses techniques popularized by Natural Language Processing to analyze public playlists and record what songs/artists often appear together.

To complement the existing algorithms of Spotify, Camacho-Horvitz and de Leon (2016) analyzed the similarity of songs based on metadata about each song. First, the authors set up a similarity score predictor which uses linear regression on simple metrics from the Million Song Dataset and Spotify Web API information. Then, the model they used to generate playlists was framed using weight- maximization problem, where except for the weight of the song from the similarity score predictor, user preferences can also allow users to eliminate certain possible playlists.

He, Li, and Nguy (2015) utilize song similarity through network analysis, community detection and Personalized PageRank, to develop playlist generation. First, the song similarity network is casted as a directed graph using song similarity scores from Spotify. To identify the community network, they use the Louvain algorithm, a method based on greedy algorithm to optimize the sub-parts of the network. Lastly, page rank is used to sequence the playlist.

Lastly, an example of neural networks used in playlist generation is from He, Liao and Zhang (2017). They explore neural network architectures for collaborative filtering. According to their work, a key factor in collaborative filtering is the interaction between user and item features. Therefore, they used matrix factorization and applied an inner product on the features of users and items. In order to learn the arbitrary function from data, the authors replace the inner product with a multilayered neural architecture, a general framework named NCF, short for Neural network based Collaborative Filtering. Their evidence shows that using deeper layers of neural networks offers better recommendation performance.

### References

Camacho-Horvitz, Miguel, and Christopher Ponce de Leon (2016). *Playlist Generation Using Song Similarity Scoring*. Stanford University, http://web.stanford.edu/class/cs221/2017/restricted/p-final/somosmas/final.pdf. 

He, B., Li, Y., & Nguy, B. (2015). *Music Playlist Generation based on Community Detection and Personalized PageRank*. Stanford. Retrieved from http://snap.stanford.edu/class/cs224w-2015/projects_2015/Music_Playlist_Generation.pdf

He, X. , Liao, L. , Zhang, H., *Neural Collaborative Filtering, Proceedings of the 26th International Conference on World Wide Web*, April 03-07, 2017, Perth, Australia [doi>10.1145/3038912.3052569]