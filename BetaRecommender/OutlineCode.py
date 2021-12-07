#Import Files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

#Each element of Ratings includes userID, gameID, and rating
ratings = pd.read_csv('ratings.csv')
ratings.head()

#Each element of Games includes gameID, title, genres/features
games = pd.read_csv("movies.csv")
games.head()

unique_user = ratings.userId.nunique(dropna = True)
unique_game = ratings.gameId.nunique(dropna = True)
print("number of unique user:")
print(unique_user)
print("number of unique games:")
print(unique_game)

# for creating item user matrix  .. we need to check how many ratings we have here or how many are absent .
total_ratings = unique_user*unique_movie
rating_present = ratings.shape[0]

ratings_not_provided = total_ratings - rating_present 

print("ratings not provided means some user have not watched some movies and its given by")
print(ratings_not_provided)
print("sparsity of user-item matrix is :")
print(ratings_not_provided / total_ratings)



user_cnt = pd.DataFrame(ratings.groupby('userId').size(),columns=['count'])
user_cnt_copy = user_cnt
user_cnt.head()


# we have to reshape/prepare our dataset into a format which can be given as parameter. 
# we will pivot our final dataset into a ITEM-USER matrix and empty cell with 0
#After culling unrated games
final_ratings = ratings_with_popular_games_with_active_user
#final_ratings.shape
item_user_mat = final_ratings.pivot(index='gameID',columns = 'userId',values='rating').fillna(0)

# create a mapper which maps movie index and its title
game_to_index = {
    game:i for i,game in enumerate(list(games.set_index('gameID').loc[item_user_mat.index].title))
}
# create a sparse matrix for more efficient calculations
from scipy.sparse import csr_matrix
item_user_mat_sparse = csr_matrix(item_user_mat.values)

#. now when a game name is given as input we need to find if that game is present in our dataset or not.
#. If it is not present then we cant recommend anything . so for string matching we are going to use fuzzy matching , 
#. based on result of fuzzy matching , a list of recommedation will be generated. 
#. lets create a function which take parameters (input_string , mapper=game_to_index) . 
#. This fucntion will return gameID of game title which is best match with input string . It also prints the all matches .
pip install fuzzywuzzy

# fuzzy_movie_name_matching
from fuzzywuzzy import fuzz

def fuzzy_game_name_matching (input_str,mapper,print_matches):
    # match_movie is list of tuple of 3 values(game_name,index,fuzz_ratio)
    match_game = []
    for game,ind in mapper.items():
        current_ratio = fuzz.ratio(game.lower(),input_str.lower())
        if(current_ratio>=50):
            match_game.append((game,ind,current_ratio))

    # sort the match_game with respect to ratio 

    match_game = sorted(match_game,key =lambda x:x[2])[::-1]

    if len(match_game)==0:
        print("Oops..! no such game is present here\n")
        return -1
    if print_matches == True:
        print("some matching of input_str are\n")
        for title,ind,ratio in match_game:
            print(title,ind,'\n')


    return match_game[0][1]    

# define the model
from sklearn.neighbors import NearestNeighbors
recommendation_model = NearestNeighbors(metric='cosine',algorithm = 'brute',n_neighbors=20,n_jobs=-1)

# Recommendation function takes game name and returns recommendation
def make_recommendation(input_str,data,model,mapper,n_recommendation):
    print("system is working....\n")
    model.fit(data)

    index = fuzzy_game_name_matching (input_str,mapper,print_matches = False)

    if index==-1 :
        print("pls enter a valid game name\n")
        return 

    index_list = model.kneighbors(data[index],n_neighbors=n_recommendation+1,return_distance=False)
    # now we ind of all recommendation
    # build mapper index->title
    index_to_game={
        ind:game for game,ind in mapper.items()
    }

    print("Viewer who watches this game ",input_str,"also watches following games.")
    #print(index_list[0][2])
    for i in range(1,index_list.shape[1]):
        print(index_to_game[index_list[0][i]])



    return 
