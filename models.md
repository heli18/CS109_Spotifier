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
### Linear Regression
### Artificial Neuronetwork
### Combined Model