import spotipy
import os
from spotipy.oauth2 import SpotifyOAuth,SpotifyClientCredentials

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep

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
for i in tqdm(range(len(new_tracks))):
     popularity_list.append(sp.artist(new_tracks.iloc[i]['artist_uri'])['popularity'])
     sleep(0.02)
new_tracks['popularity'] = popularity_list

new_tracks.to_csv('../data/tracks_with_popularity.csv', index = False)
