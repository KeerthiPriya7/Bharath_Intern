import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate, train_test_split
from surprise.accuracy import rmse
from surprise import accuracy

data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.25)
sim_options = {
    'name': 'cosine',
    'user_based': False
}

model = KNNBasic(sim_options=sim_options)
model.fit(trainset)
predictions = model.test(testset)
rmse_score = accuracy.rmse(predictions)
print(f'RMSE: {rmse_score}')
def get_top_n_recommendations(predictions, n=10):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n
user_id = str(196)
top_n_recommendations = get_top_n_recommendations(predictions, n=10)
print(f'Top 10 recommendations for User {user_id}:')
for movie_id, estimated_rating in top_n_recommendations[user_id]:
    print(f'Movie ID: {movie_id}, Estimated Rating: {estimated_rating}')
