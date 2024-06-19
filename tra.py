import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import euclidean_distances

#读取训练集数据和测试集数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#只使用部分数据集进行训练
train_data = train_data.sample(frac=0.0001)  # 这里我们只使用部分数据进行训练

#将数据转换为稀疏矩阵
train_data['rating'] = 1
user_item_matrix = train_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
user_item_matrix_csr = csr_matrix(user_item_matrix.values)

# 使用TruncatedSVD进行降维
svd = TruncatedSVD(n_components=10)
user_item_matrix_pca = svd.fit_transform(user_item_matrix_csr)

#计算用户之间的相似度
user_similarity = euclidean_distances(user_item_matrix_pca)

#将相似度矩阵转换为稀疏矩阵
user_similarity_sparse = csr_matrix(user_similarity)

#为每个用户找到最相似的k个用户
k = 50
top_k_similar_users = np.argsort(user_similarity_sparse.toarray(), axis=1)[:, :k]

#预测用户对未阅读过的图书的评分
def predict_rating(user_id, item_id):
    if user_id - 1 >= len(top_k_similar_users):
        return 0
    similar_users = top_k_similar_users[user_id - 1]
    user_ratings = user_item_matrix.iloc[similar_users]
    item_ratings = user_ratings[item_id].values
    if len(item_ratings[np.nonzero(item_ratings)]) == 0:
        return 0
    else:
        mean_rating = np.nanmean(item_ratings[np.nonzero(item_ratings)])
        return mean_rating

recommendations = []

#为测试集中的每个用户生成推荐书籍列表
for user_id in test_data['user_id']:
    if user_id in user_item_matrix.index:
        unrated_items = np.setdiff1d(user_item_matrix.columns, user_item_matrix.loc[user_id].dropna().index)
    else:
        unrated_items = np.setdiff1d(user_item_matrix.columns,user_item_matrix.get(user_id, pd.Series(dtype='float64')).dropna().index)

    item_scores = [predict_rating(user_id, item_id) for item_id in unrated_items]
    sorted_items = sorted(zip(unrated_items, item_scores), key=lambda x: x[1], reverse=True)
    recommended_items = [item for item, score in sorted_items[:10]]
    recommendations.extend([(user_id, item) for item in recommended_items])

#将推荐结果保存到submission.csv文件中
recommendations_df = pd.DataFrame(recommendations, columns=['User_id', 'Item_id'])
recommendations_df.to_csv('submission.csv', index=False)
