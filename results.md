[SPOTIFIER](https://heli18.github.io/CS109_Spotifier/) |
[Introduction and EDA](https://heli18.github.io/CS109_Spotifier/intro) |
[Literature Review and Related Work](https://heli18.github.io/CS109_Spotifier/lit) |
[Models](https://heli18.github.io/CS109_Spotifier/models) |
[Results and Conclusion](https://heli18.github.io/CS109_Spotifier/results) |
[Downloads](https://heli18.github.io/CS109_Spotifier/downloads) 

# SPOTIFIER! : CS109 DATA SCIENCE FINAL PROJECT

## Results and Conclusion

### [1. Summary](#summary)
### [2. Results](#results)
#### [a. Baseline Model](#baseline-model)
#### [b. Linear Regression](#linear-regression)
#### [c. Aritificial Neuronetwork](#arificial-neuronetwork)
#### [d. Combined Model (baseline and neuronetwork)](#combined-model)

### Summary
First, we split the sample dataset into a Playlist Bank and a Training set. For each playlist in the Training Set, we hide one track to be the track that we try to predict, and we set the rest as our predictors.

To achieve the final playlist generation model, we combined a Neural Network model with our baseline model. The baseline model recommendation list includes a list of top 5 playlists with the largest similarity score to the popular tracks. However, in the combined model we do not randomly choose 100 tracks but run the Artificial Neural Network to rank the top 100 tracks with the highest interaction score and use them as our final recommendation list. In this model, we successfully include the hidden track (the track we want to predict) 26.0 percent of time.

Our final model improves on the previous model performance (linear regression, Lasso regression, and Gradient Boosted Trees), by improving the predictive performance or in the case of GBT, primarily addressing the issue of overfitting identified in them. Below we elaborate on the models used and conclusions from each one of them.

![results_01](https://heli18.github.io/CS109_Spotifier/images/Results1.JPG)

### Results

#### Baseline Model
The baseline model is based on comparing each pair of “tracks” and hence computing the “similarity score” by using the 'Jaccard similarity' similarity measurement:The dataset we use in baseline model is the full dataset(10,000) playlist.

![results_02](https://heli18.github.io/CS109_Spotifier/images/Results2.JPG)

For each playlist in the Training Set, we hide one track to be the track that we try to predict, and we set the rest as our predictors. For each training list, if the hidden track exists in our prediction track list, we will define the error rate of this training list to 0, otherwise, to 1. And we average the total error of 500 playlists to be our total error rate.

For the final recommendation tracks, we will first get the not-overlap tracks in top 5 similary playlists, and randomly select 100 as our recommendation.

The score of the model is 0.166. The main issues identified with it include potential bias, correlation, and computation cost. 

#### Linear Regression
Linear regression is the first step in the improvement we made to our baseline model. The response variable chosen is num_followers, as the songs in a playlist with a high number of followers are most likely to be popular or become popular in the future. The predictors used were only the numerical variables and removing the other categorical variables. So the predictors included were (collaborative, duration, num_albums, num_artists, num_edits, num_tracks, duration_min).

The score for this model is -0.0038. 

The first issue we identified with this model were the outliers. The top playlists are called ‘Now Playing’ and ‘Top Pop’ which is one of Spotify and Interscope generated playlists. Most likely, these songs are predefined to already be popular, or up and coming. However, we decide to keep these playlists in the dataset due the large number of followers that they have and their impact in the model performance and our dependent variable (num_followers).

##### Linear Regression - hot encoded "popular artist"

The next linear regression model included all of the variables from the previous model but also included the ‘popular artist variable which was hot encoded. We saw improvement to the model, with new score equal 0.037

##### Lasso

We also ran a Lasso model with the regularization parameter equal to 0.5 which showed that the number of parameters is not an issue in our linear regression model, since our accuracy score with LASSO was -0.43.

##### Gradient Boosted Trees

Our Gradient Boosted Model included a max depth of 10, learning rate equal to 0.1 and 60 estimators. The accuracy score of this model on train data is 0.67, however the score is negative on test data, -11.1478, which shows that is having overfitting issues. The large number of trees can be one problem with it. 

#### Arificial Neuronetwork

Our artificial neural network is based on He et al (2017). The 1 Million playlist dataset was converted into a matrix with rows presenting playlists and columns of the matrix presenting tracks.

We put the data in the matrix format illustrated below. Each row of the matrix represents the user (playlist) and each column of matrix represent the items (tracks). The value of the matrix is the "interaction". We mark "1" as the interaction of user and item exist and 0 otherwise. 

![results_03](https://heli18.github.io/CS109_Spotifier/images/Results3.JPG)

Matrix Factorization (MF) associates each user and item with a real-valued vector of latent features. It is shown below. The latent features are presented in the right handside of the above figures as compared with the matrix. As the research of Neural Collaborative Filtering indicated, there are some differences between "Jaccard similarity" and the "matrix factorization": in matrix, the u4 is most similar to u1, followed by u3 and u2 while in MF, the P4 is similar to P1, followed by P2 and P3 which indicated the "ranking lost" of MF. This could be one of the sources of the model limitations.

![results_04](https://heli18.github.io/CS109_Spotifier/images/Results4.JPG)

Our neural network aimed to estimate the function to predict the y_hat(ui). The input layer includes two datasets: the "user" and "track". The second layer is the embedding layer with user latent vector and tracks latent vector. And we added other layers in neural network to tune our predictions. The output layer is the predicted "interaction" ('y' in dataset). The prediction score is a probability ranged from 0 to 1. The probability indicated the likelihood of the interactions between the playlist and track. The higher the probability, the more likely that the user will like the track.

![results_05](https://heli18.github.io/CS109_Spotifier/images/Results5.JPG)

#### Combined Model

Similar to the initial stages, we hide one track in each playlist(user) and generate a recommendation list, see if this recommendation list contains that hidden track. However, due to our limitation of training data size and the very large tracks in the music bank, we don't expect that for a large percent of time, we can successfully find out that exact one hidden track. In real world, the user preference can be much broader than just one "hidden" track.

We choose to combine the baseline model and Neural Network model together to generate the list. First, we will extract our previous recommendation list (including the not overlap tracks of top 5 similar playlists in baseline model). Instead of randomly select 100 of them as recommendation, we apply the trained Artifical Neural Network to rank the possible interactions and select the top 100 tracks with the highest interaction score and use them as our final recommendation list.

As far as our prediction score increase (higher than what we get in baseline model), our neural network works. 26.0 percent of time, we successfully include the hidden track. 

Just as what we have expected, even though the prediction accuracy of neural network is very high, the probability that the recommendation list we generate that will include the hidden track is very low because: The hidden track is not the exact indicator of "user preference" because the user may have higher probability to like the other tracks that does not necessarily to be the hidden track! Hence the real "user preference" for the not hidden track in other playlists will be higher than the actual "hidden track" that we hided before. But in the scope of this project, we are not able to find out the true user preference and hence, we can just use the "hidden track" as a possible indicator.

Therefore, even though our score is not very high, it did improve a lot comparing to our baseline model, which is basically in random basis. This is consistent with our high neural network prediction results.