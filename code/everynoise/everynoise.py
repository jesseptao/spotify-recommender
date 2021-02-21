import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_new_songs(url):
    # looking at HTML structure of everynoise new releases page
    home_url = url
    res = requests.get(home_url)
    soup = BeautifulSoup(res.content, 'lxml')

    # finding all rows with artist and song name with uris
    album_row = soup.find_all('div', attrs = {"class": "albumrow"})

    # creating a list of dictionaries to create a DataFrame
    tracks = []
    for i in range(len(album_row)):
        if len(album_row[i].find_all('span', attrs = {"class": "trackcount"})) == 0:
            indiv_track = [{'artist_uri': album_row[i].find_all('a')[0]['href'],
                          'artist': album_row[i].find_all('a')[0].find_all('b')[0].text,
                          'track_uri': album_row[i].find_all('span')[0]['trackid'],
                          'track_name': album_row[i].find_all('a')[1].text}]
            tracks += indiv_track
    tracks = pd.DataFrame(tracks)
    return tracks
