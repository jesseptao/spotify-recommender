# import library
from everynoise import get_new_songs

import spotipy
import os
from spotipy.oauth2 import SpotifyOAuth,SpotifyClientCredentials

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import numpy as np

# get new song
tracks = get_new_songs('http://everynoise.com/new_releases_by_genre.cgi?genre=anygenre&region=US')

# save to csv to use later
tracks.to_csv('../data/new_tracks.csv', index = False)

# load in csv
new_tracks = pd.read_csv('../data/new_tracks.csv')

# create track_url column
new_tracks['track_url'] = new_tracks['track_uri'].replace('spotify:track:',
                                                          'https://open.spotify.com/track/',
                                                          regex = True).astype(str)

# setup spotify with app credentials, use environment variables so GitHub scrapers don't get access to my spotify developer keys
cid = os.getenv('SPOTIPY_CLIENT_ID')
secret = os.getenv('SPOTIPY_CLIENT_SECRET')

#https://medium.com/@maxtingle/getting-started-with-spotifys-api-spotipy-197c3dc6353b

client_credentials_manager = SpotifyClientCredentials(client_id = cid,
                                                      client_secret = secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

popularity_list = []
genre_list = []
for i in tqdm(range(len(new_tracks))):
     popularity_list.append(sp.artist(new_tracks.iloc[i]['artist_uri'])['popularity'])
     genre_list.append(sp.artist(new_tracks.iloc[i]['artist_uri'])['genres'])
     sleep(0.13)
new_tracks['popularity'] = popularity_list
new_tracks['artist_genres'] = genre_list

new_tracks.to_csv('../data/tracks_with_popularity.csv', index = False)

new_tracks_popularity = pd.read_csv('../data/tracks_with_popularity.csv')

# drop songs by artists with more than 70 popularity
new_tracks_popularity = new_tracks_popularity[new_tracks_popularity['popularity'] <= 70].copy()

# only need to get user features after getting new_track features once
new_features_list = []
for i in tqdm(range(len(new_tracks))):
    try:
        new_features_list.append(sp.audio_features(new_tracks_popularity.iloc[i]['track_uri'])[0])
        sleep(0.067)
    except:
        sleep(0.25)

def is_empty(any_structure):
    if any_structure:
        return False
    else:
        print('Structure is empty.')
        return True

new_features_list = [i for i in new_features_list if is_empty(i) == False]
new_features_df = pd.DataFrame(new_features_list)
new_features_df['artist'] = [i['artist'] for i in new_tracks_popularity for j in new_features_df if i['track_uri'] == j['uri']]
new_features_df['genre'] = [i['artist_genres'] for i in new_tracks_popularity for j in new_features_df if i['track_uri'] == j['uri']]
new_features_df['track_name'] = [i['track_name'] for i in new_tracks_popularity for j in new_features_df if i['track_uri'] == j['uri']]
new_features_df.to_csv('../data/new_track_features.csv', index = False)
