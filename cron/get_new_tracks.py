# import library
from everynoise import get_new_songs

# get new song
tracks = get_new_songs('http://everynoise.com/new_releases_by_genre.cgi?genre=anygenre&region=US')

# save to csv to use later
tracks.to_csv('../data/new_tracks.csv', index = False)
