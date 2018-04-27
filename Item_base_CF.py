#基于物品的协同过滤算法实现
#create : 2018-3-18
#author: wenyi

#基于物品的协同过滤
class ItemBaseCF(object):
    """
    this code is the base item CF
    """
    def __init__(self, data):
        """
        data: numpy array the data is the input data include user,item and rating
        """
        self.data = data
        #dict for mid,rat in self.user_movie[uid].items()
        self.movie_user = dict()   
        #dict for item movie
        self.user_movie = dict()   
        for i in range(data.shape[0]):
            uid = data[i][0]
            mid = data[i][1]
            rat = data[i][2]
            self.movie_user.setdefault(mid, {})
            self.user_movie.setdefault(uid, {})
            self.movie_user[mid][uid] = rat
            self.user_movie[uid][mid] = rat 
        self.similarity = dict()

    def cal_similarity(self, m1, m2):
        """
        m1,m2:movie id
        """
        #在两个movie id 中共现的user
        co_user = []
        for user in self.movie_user[m1]:
            if user in self.movie_user[m2]:
                co_user.append(user)

        Nm1 = 0.0
        Nm2 = 0.0
        res = 0.0
        #m1历史看过的uid及评分情况
        for u, r in self.movie_user[m1].items():
            Nm1 += r*r
        #m2历史看过的uid及评分情况
        for u1, r1 in self.movie_user[m2].items():
            Nm2 +=  r1*r1

        #m1和m2历史看过的uid的评分情况
        for u in co_user:
            res += self.movie_user[m1][u] * self.movie_user[m2][u]
        return res/np.sqrt(Nm1*Nm2)

    def train(self):
        movies = set(self.data[:,1])

        #计算数据集中任意两个movie的相似度
        for m1 in movies:
            for m2 in movies:
                self.similarity.setdefault(m1, {})
                self.similarity.setdefault(m2, {})
                if m1 == m2:
                    self.similarity[m1][m2] = 1
                else: 
                    self.similarity[m1][m2] = self.cal_similarity(m1, m2)
                    self.similarity[m2][m1] = self.similarity[m1][m2]

    def predict(self, uid, mid,K=5):
        """
        uid: id is the user id of predict data
        mid: id is the movie id of predict data
        return: rating of uid and mid 
        """
        #预测某个用户和对某个电影的评分
        recommend_movie = list()
        for m,rat in self.user_movie[uid].items():
            if self.similarity[mid][m] > 0 and mid not in self.user_movie[uid]:
                recommend_movie.append((m,self.similarity[mid][m]*rat))
        recommend_movie = sorted(recommend_movie,key=lambda x:x[1],reverse=True)
        if len(recommend_movie) > K:
            sums = 0
            for mid,rat in recommend_movie[:K]:
                sums += rat
            return sums/K
        else:
            sums = 0
            for mid,rat in recommend_movie:
                sums += rat
            return sums/len(recommend_movie)
    
    def recomend_movie(self,uid,K=10):
        """
        给某个uid推荐电影,得到推荐候选集
        """
        #recommend list
        recommend_list = {}
        #uid历史看过的评分大于三的电影
        self.movie_list = []
        for mid,rat in self.user_movie[uid].items():
            if rat > 3:
                self.movie_list.append((mid,rat))
        for mid,rat in self.movie_list:
            sim_movie = sorted(self.similarity[mid].items(), key=lambda x:x[1],reverse=True)
            for m,sim in sim_movie:
                recommend_list.setdefault(m,0)
                recommend_list[m] += sim*rat
        return sorted(recommend_list.items(),key=lambda x:x[1],reverse=True)

    #推荐理由
    def recomend_reason(self,recommend_list,movies,K=3):
        """
        给出推荐的原因:推荐K部电影
        movies: this is a table of movies and titles
        """
        reason_list = []
        
        for m,rat in self.movie_list:
            d = {}
            h_title = movies[movies['MovieID']==m]['Title'].values
            for mid, rat in recommend_list:
                if self.similarity[mid][m]>0:
                    if mid not in d:
                        d[mid] = self.similarity[mid][m]
            if len(d)>K:
                for m, sim in sorted(d.items(),key=lambda x:x[1], reverse=True)[1:K+1]:
                    r_title = movies[movies['MovieID']==m]['Title'].values
                    print("because you have watched the movie %s so recommend movie %s to you the similarity is %.5f" %(h_title, r_title, sim))
            else:
                for m, sim in sorted(d.items(),key=lambda x:x[1], reverse=True)
                r_title = movies[movies['MovieID']==m]['Title'].values
                    print("because you have watched the movie %s so recommend movie %s to you the similarity is %.5f" %(h_title, r_title, sim))


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table('ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine = 'python')
    users_title = ['UserID','Gender','Age','Occupation','Zip-code']
    users = pd.read_table('ml-1m/users.dat',sep='::',header=None,names=users_title,engine='python')
    rating_title = ['UserID','MovieID','Rating','Timestamp']
    ratings = pd.read_table('ml-1m/ratings.dat',sep='::',header=None,names=rating_title,engine='python')
    data = pd.merge(pd.merge(ratings,users),movies)
    
    #数据集太大,计算很慢,这里取前1000个数据进行计算
    data = data[:10000]
    data = data.iloc[:,:3].values
    ibc = ItemBaseCF(data)
    ibc.train()

    #uid =6时
    recommend_list = ibc.recomend_movie(6)
    ibc.recomend_reason(recommend_list,movies)

    #uid = 12时
    recommend_list = ibc.recomend_movie(12)
    ibc.recomend_reason(recommend_list,movies)