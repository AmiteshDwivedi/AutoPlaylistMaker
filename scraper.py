import spotipy
import pandas as pd
import numpy as np
import time
from main import initSpotipy, artist_tracks_dict, create_top_tracks_dict, create_features, add_to_playlist, relevant_features, all_similarities, all_distance, extend_frame

def get_similar_songs_by_artist(sp, artist_name, max_recommendations=5):
    artist = artist_tracks_dict(artist_name)
    a_df = pd.DataFrame(artist)
    return find_similar_tracks(a_df, sp, max_recommendations)

def get_similar_songs_from_spotify(sp, max_recommendations=5):
    lt = create_top_tracks_dict(sp, 'long_term')
    mt = create_top_tracks_dict(sp, 'medium_term') 
    st = create_top_tracks_dict(sp, 'short_term')
    
    lt_features = create_features(lt)
    mt_features = create_features(mt)
    st_features = create_features(st)
    
    lt_df = pd.DataFrame(lt)
    mt_df = pd.DataFrame(mt)
    st_df = pd.DataFrame(st)
    
    lt_df = extend_frame(lt_df)
    mt_df = extend_frame(mt_df)
    st_df = extend_frame(st_df)
    
    all_df = pd.concat([lt_df, mt_df, st_df])
    
    return find_similar_tracks(all_df, sp, max_recommendations)

def find_similar_tracks(tracks_df, sp, max_recommendations):
    features = relevant_features(tracks_df)
    
    tracks_df.key = tracks_df.key/11
    
    profile = np.array([tracks_df[feat].mean() for feat in features])
    compare = tracks_df[[features[0],features[1],features[2]]].values

    all_similarities(profile, compare, tracks_df)
    tracks_df = tracks_df.nlargest(max_recommendations, 'cos_similarity')

    all_distance(profile, compare, tracks_df)
    recommendations = tracks_df.drop_duplicates('track_name').nsmallest(max_recommendations, 'euclidean')
    
    add_to_playlist(sp, recommendations)
    return recommendations

if __name__ == "__main__":
    sp = initSpotipy()
    option = input("Choose an option:\n1. Find similar songs by artist\n2. Find similar songs from Spotify\nYour choice: ")

    if option == '1':
        artist_name = input("Enter artist name: ")
        recommendations = get_similar_songs_by_artist(sp, artist_name)
    elif option == '2':
        recommendations = get_similar_songs_from_spotify(sp)
    else:
        print("Invalid option. Exiting.")
        sys.exit(1)

    print(f"Generated playlist with {len(recommendations)} tracks.")
