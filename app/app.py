"""
Prerequisites
    pip3 install spotipy Flask Flask-Session
    // from your [app settings](https://developer.spotify.com/dashboard/applications)
    export SPOTIPY_CLIENT_ID=client_id_here
    export SPOTIPY_CLIENT_SECRET=client_secret_here
    export SPOTIPY_REDIRECT_URI='http://127.0.0.1:8080' // must contain a port
    // SPOTIPY_REDIRECT_URI must be added to your [app settings](https://developer.spotify.com/dashboard/applications)
    OPTIONAL
    // in development environment for debug output
    export FLASK_ENV=development
    // so that you can invoke the app outside of the file's directory include
    export FLASK_APP=/path/to/spotipy/examples/app.py
 
    // on Windows, use `SET` instead of `export`
Run app.py
    python3 -m flask run --port=8080
    NOTE: If receiving "port already in use" error, try other ports: 5000, 8090, 8888, etc...
        (will need to be updated in your Spotify app and SPOTIPY_REDIRECT_URI variable)
"""

import os
from flask import Flask, session, request, redirect, render_template
from flask_session import Session
import spotipy
import uuid
import json
import numpy as np
import pandas as pd
from time import sleep
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(64)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
Session(app)

caches_folder = './.spotify_caches/'
if not os.path.exists(caches_folder):
    os.makedirs(caches_folder)

new_features_df = pd.read_csv('../data/new_track_features.csv')

def session_cache_path():
    return caches_folder + session.get('uuid')

def generate_user_tracks(user_tracks):
    user_df = pd.DataFrame(user_tracks['items'])
    track_url = []
    track_id = []
    track_name = []
    artist_uri = []
    artist_name = []
    track_uri = []
    popularity = []

    for i in range(len(user_df)):
        track_url.append(user_df.iloc[i]['track']['href'])
        track_name.append(user_df.iloc[i]['track']['name'])
        track_uri.append('spotify:track:' + user_df.iloc[i]['track']['id'])
        artist_uri.append('spotify:artist:' + user_df.iloc[i]['track']['artists'][0]['id'])
        artist_name.append(user_df.iloc[i]['track']['artists'][0]['name'])
    user_df['track_url'] = track_url
    user_df['track_uri'] = track_uri
    user_df['track_name'] = track_name
    user_df['artist_uri'] = artist_uri
    user_df['artist'] = artist_name

    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = spotipy.oauth2.SpotifyOAuth(cache_handler=cache_handler)
    spotify = spotipy.Spotify(auth_manager = auth_manager)

    for i in range(len(user_df)):
        popularity.append(spotify.artist(user_df.iloc[i]['artist_uri'])['popularity'])
        sleep(0.02) 
    user_df['popularity'] = popularity

    return user_df

def compute_distance(a, b):
    normalize_a = tf.nn.l2_normalize(a, 1)
    normalize_b = tf.nn.l2_normalize(b, 1)
    distance = 1 - tf.matmul(normalize_a, normalize_b, transpose_b = True)
    return distance

def is_empty(any_structure):
    if any_structure:
        return False
    else:
        print('Structure is empty.')
        return True

@app.route('/')
def index():
    if not session.get('uuid'):
        # Step 1. Visitor is unknown, give random ID
        session['uuid'] = str(uuid.uuid4())

    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = spotipy.oauth2.SpotifyOAuth(scope='user-read-recently-played',
                                                cache_handler=cache_handler, 
                                                show_dialog=True)

    if request.args.get("code"):
        # Step 3. Being redirected from Spotify auth page
        auth_manager.get_access_token(request.args.get("code"))
        return redirect('/')

    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        # Step 2. Display sign in link when no token
        auth_url = auth_manager.get_authorize_url()
        return render_template('login.html', auth_url = auth_url)

    # Step 4. Signed in, display data
    spotify = spotipy.Spotify(auth_manager=auth_manager)
    user_name = spotify.me()["display_name"]
    return render_template('user_main.html', user_name = user_name)

@app.route('/sign_out')
def sign_out():
    try:
        # Remove the CACHE file (.cache-test) so that a new user can authorize.
        os.remove(session_cache_path())
        session.clear()
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
    return redirect('/')
    
@app.route('/recently_played')
def recently_played():
    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = spotipy.oauth2.SpotifyOAuth(cache_handler=cache_handler)
    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        return redirect('/')

    spotify = spotipy.Spotify(auth_manager=auth_manager)
    recently_played = spotify.current_user_recently_played()
    with open(f'/media/jesse/Number3/json/{spotify.me()["id"]}.json', 'w') as outfile:
        json.dump(recently_played, outfile)
    return render_template('recently_played.html', recently_played = recently_played["items"])

@app.route('/recommendations')
def recommendations():
    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = spotipy.oauth2.SpotifyOAuth(cache_handler=cache_handler)
    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        return redirect('/')

    spotify = spotipy.Spotify(auth_manager=auth_manager)
    with open(f'/media/jesse/Number3/json/{spotify.me()["id"]}.json') as read:
        user_tracks = json.load(read)
    user_df = generate_user_tracks(user_tracks)
    user_features_list = []
    for i in range(len(user_df)):
        user_features_list.append(spotify.audio_features(user_df.iloc[i]['track_uri'])[0])
        sleep(0.02)
    user_features_list = [i for i in user_features_list if is_empty(i) == False]
    user_features_df = pd.DataFrame(user_features_list)
    combined_features_df = pd.concat([new_features_df, user_features_df])
    combined_features_df.reset_index(drop = True, inplace = True)
    combined_features_df.drop(['type', 'id', 'duration_ms', 'time_signature', 'track_href',
                  'analysis_url'], axis = 1, inplace = True)
    compare_df = combined_features_df[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'mode']]
    mms = MinMaxScaler()
    compare_df_sc = mms.fit_transform(compare_df)
    compare_df_sc = pd.DataFrame(compare_df_sc, columns = compare_df.columns)
    distances = compute_distance(compare_df_sc, compare_df_sc)
    distances_df = pd.DataFrame(distances.numpy(), index = combined_features_df['uri'], columns = combined_features_df['uri'])
    distances_df.loc['score'] = distances_df.tail(len(user_features_list)).sum()
    similar_5 = distances_df.loc['score'][:-len(user_features_list)].sort_values()[0:5].index
    unsimilar_5 = distances_df.loc['score'][:-len(user_features_list)].sort_values(ascending = False)[0:5].index
    similar_track_names = []
    unsimilar_track_names = []
    distances_df.loc['score'] = 0
    for i in similar_5:
        similar_track_names.append(spotify.track(i))
    for i in unsimilar_5:
        unsimilar_track_names.append(spotify.track(i))

    return render_template('recommendations.html', user_df = user_df, similar_track_names = similar_track_names, unsimilar_track_names = unsimilar_track_names)
'''
Following lines allow application to be run more conveniently with
`python app.py` (Make sure you're using python3)
(Also includes directive to leverage pythons threading capacity.)
'''
if __name__ == '__main__':
	app.run(threaded=True, port=int(os.environ.get("PORT", 8090)))
