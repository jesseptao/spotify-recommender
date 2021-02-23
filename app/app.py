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
from spotipy.oauth2 import SpotifyClientCredentials
import uuid
import json
import numpy as np
import pandas as pd
from time import sleep
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from flask_mysqldb import MySQL
from rq import Queue
from rq.job import Job
from worker import conn

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(64)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
app.config['MYSQL_HOST'] = '192.168.1.221'
app.config['MYSQL_USER'] = 'yasr'
app.config['MYSQL_PASSWORD'] = 'ILoveDSI-!!!^'
app.config['MYSQL_DB'] = 'yasr'
Session(app)

mysql = MySQL(app)
q = Queue(connection=conn)

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

    #cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = SpotifyClientCredentials()
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

def get_recommendations(user_id):
    #cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    #auth_manager = spotipy.oauth2.SpotifyOAuth(cache_handler=cache_handler)
    #if not auth_manager.validate_token(cache_handler.get_cached_token()):
    #    return redirect('/')

    auth_manager = SpotifyClientCredentials()
    spotify = spotipy.Spotify(auth_manager = auth_manager)
    with open(f'/media/jesse/Number3/json/{user_id}.json') as read:
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

    return [similar_track_names, unsimilar_track_names]

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

@app.route('/prepare_recommendations', methods=['POST'])
def prepare_recommendations():
    if request.method == "POST":
        from app import get_recommendations
        cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
        auth_manager = spotipy.oauth2.SpotifyOAuth(cache_handler=cache_handler)
        if not auth_manager.validate_token(cache_handler.get_cached_token()):
            return redirect('/')

        spotify = spotipy.Spotify(auth_manager=auth_manager)
        user_id = spotify.me()["id"]
        job = q.enqueue(get_recommendations, args=(user_id,), result_ttl = 900)
        position = len(q)
        return render_template('prepare_recommendations.html', position = position + 1, id = job.id)


@app.route('/recommendations/<job_key>', methods = ['GET'])
def recommendations(job_key):
    job = Job.fetch(job_key, connection = conn)
    job_ids_list = q.job_ids
    while not (job.is_finished):
        if job_key in job_ids_list:
            position = job_ids.index(job_key)
        else:
            position = 0
        return render_template('prepare_recommendations.html', position = position + 1, id = job.id)
        sleep(2)
    results = job.result
    return render_template('recommendations.html', result = results)

@app.route('/submit', methods=["GET", "POST"])
def submit():
    if request.method == "POST":

        cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
        auth_manager = spotipy.oauth2.SpotifyOAuth(cache_handler=cache_handler)
        if not auth_manager.validate_token(cache_handler.get_cached_token()):
            return redirect('/')

        spotify = spotipy.Spotify(auth_manager=auth_manager)
        user_id = spotify.me()["id"]
        score = request.form['score']
        score = int(score)
        print(type(score))
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO similar_scores(user_id, scores) VALUES (%s, %s)", (user_id, score))
        mysql.connection.commit()
        cur.close()
        return render_template('submit.html')
    
    return render_template('submit.html')

@app.route('/submit_unfamiliar', methods=["GET", "POST"])
def submit_unfamiliar():
    if request.method == "POST":

        cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
        auth_manager = spotipy.oauth2.SpotifyOAuth(cache_handler=cache_handler)
        if not auth_manager.validate_token(cache_handler.get_cached_token()):
            return redirect('/')

        spotify = spotipy.Spotify(auth_manager=auth_manager)
        user_id = spotify.me()["id"]
        score = request.form['score']
        score = int(score)
        print(type(score))
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO unsimilar_score(user_id, scores) VALUES (%s, %s)", (user_id, score))
        mysql.connection.commit()
        cur.close()
        return render_template('submit.html')
    
    return render_template('submit.html')

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

'''
Following lines allow application to be run more conveniently with
`python app.py` (Make sure you're using python3)
(Also includes directive to leverage pythons threading capacity.)
'''
if __name__ == '__main__':
	app.run(threaded=True, port=int(os.environ.get("PORT", 8090)))
