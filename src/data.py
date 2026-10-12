import torch
import random
import pandas as pd
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

random.seed(0)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        分别接收三个张量user/item/target
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        #返回一条样本(user,item,target)
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        #返回总样本数
        return self.user_tensor.size(0)


class SampleGenerator(object):
    """构建负样本，进行训练、测试样本划分"""

    def __init__(self, ratings):
        """
        args:
            ratings需要包含以下四个元素：['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        #使用_normalize进行显式反馈，使用_binarize进行隐式反馈
        #self.preprocess_ratings = self._normalize(ratings)
        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())

        #拒绝采样需要不断从全部item中随机抽取候选item
        #提前转换为list，避免每次采样时重复执行list(self.item_pool)，开销大
        self.item_list = list(self.item_pool)

        self.negatives = self._sample_negative(ratings)
        self.train_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)

    def _normalize(self, ratings):
        """显式反馈处理：归一化，将rating由原来的[0,max_rating]放缩到[0,1]间"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings

    def _binarize(self, ratings):
        """隐式反馈处理：rating大于0，就设为1.0；rating=0，保留0"""
        ratings = deepcopy(ratings)  #复制一份数据，避免修改原始数据
        ratings.loc[ratings['rating'] > 0, 'rating'] = 1.0  #replace ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings

    def _split_loo(self, ratings):
        """leave-one-out划分训练和测试（对于每个用户，保留其最新交互作为测试集，并利用剩余的数据进行训练）"""
        #对每个用户按照timestamp降序排名，ascending=False表示时间越新，排名越前
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]  #每个用户最新的一条交互，作为测试集
        train = ratings[ratings['rank_latest'] > 1]  #其余交互，作为训练集
        assert train['userId'].nunique() == test['userId'].nunique()  #确保每个用户在训练集和测试集里都出现
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    #拒绝采样函数
    def _rejection_sample(self, interacted_items, num_negatives):
        """
        直接从全部item中随机抽取一个候选item，如果该item已经被用户交互过，
        则抽取下一个。如果没有交互过，则作为负样本保留，直到获得指定数量的负样本
        """
        #使用set保存，防止出现重复样本
        negative_samples = set()
        while (len(negative_samples) < num_negatives):
            item = random.choice(self.item_list)
            if item not in interacted_items:
                negative_samples.add(item)
        return list(negative_samples)

    def _sample_negative(self, ratings):
        """
        保存每个用户已经交互过的item集合interacted_items；
        为评估阶段生成99个固定负样本
        """
        #保存每个用户已经交互过的item集合interacted_items
        #ratings.groupby('userId')['itemId'].apply(set)：按用户分组，把每个用户交互过的所有itemId收集成一个set
        #.reset_index().rename(columns={'itemId': 'interacted_items'})：整理成[{userId,interacted_items}的数据格式]
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'}
        )

        #直接通过拒绝采样生成99个测试固定负样本
        interact_status['negative_samples'] = interact_status['interacted_items'].apply(
            lambda x: self._rejection_sample(x, 99)
        )

        #这里只需要保存用户已经交互的item和测试负样本
        return interact_status[['userId', 'interacted_items', 'negative_samples']]

    def instance_a_train_loader(self, num_negatives, batch_size):
        """instance train loader for one training epoch
           每次调用instance_a_train_loader都会重新采样，所以每个epoch的负样本可能不同，这叫动态负采样
        """
        #训练样本构建
        users, items, ratings = [], [], []

        #原来合并的是negative_items完整负样本池
        #train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        #现在只合并该用户真正交互过的interacted_items集合
        train_ratings = pd.merge(self.train_ratings,self.negatives[['userId', 'interacted_items']],on='userId')

        #每一条正样本直接通过拒绝采样生成num_negatives个负样本
        train_ratings['negatives'] = train_ratings['interacted_items'].apply(
            lambda x: self._rejection_sample(x, num_negatives)
        )

        #样本构建，itertuples()按行迭代DataFrame
        for row in train_ratings.itertuples():
            #处理正样本，生成一行正样本
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            #处理负样本，生成num_negatives行负样本
            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))  #negative samples get 0 rating
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)  #num_worker原来为默认0

    @property
    def evaluate_data(self):
        """评估样本构建"""
        #合并测试正样本和固定负样本
        test_ratings = pd.merge(self.test_ratings,self.negatives[['userId', 'negative_samples']],on='userId')
        test_users, test_items, negative_users, negative_items = [], [], [], []
        for row in test_ratings.itertuples():
            #遍历每个用户的一个测试正样本
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            #遍历该用户的99个固定负样本
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        return [torch.LongTensor(test_users),torch.LongTensor(test_items),torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)
        ]