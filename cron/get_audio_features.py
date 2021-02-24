import spotipy
from spotipy.oauth2 import SpotifyOAuth,SpotifyClientCredentials

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import sleep

# setup spotify with app credentials, use environment variables so GitHub scrapers don't get access to my spotify developer keys
cid = os.getenv('SPOTIPY_CLIENT_ID')
secret = os.getenv('SPOTIPY_CLIENT_SECRET')

#https://medium.com/@maxtingle/getting-started-with-spotifys-api-spotipy-197c3dc6353b

client_credentials_manager = SpotifyClientCredentials(client_id = cid,
                                                      client_secret = secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

# load in new tracks csv
new_tracks = pd.read_csv('../data/tracks_with_popularity.csv')

# drop songs by artists with more than 70 popularity
new_tracks = new_tracks[new_tracks['popularity'] <= 70]

# only need to get user features after getting new_track features once
new_features_list = []
for i in tqdm(range(len(new_tracks))):
    new_features_list.append(sp.audio_features(new_tracks.iloc[i]['track_uri'])[0])
    sleep(0.5)

def is_empty(any_structure):
    if any_structure:
        return False
    else:
        print('Structure is empty.')
        return True

new_features_list = [i for i in new_features_list if is_empty(i) == False]
new_features_df = pd.DataFrame(new_features_list)
new_features_df.to_csv('../data/new_track_features.csv', index = False)
