import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator

def mytrain(dim,model,dataset):
    global_latent_dim=dim
    global_model=model#0为GMF,1为MLP，2为NeuMF
    global_dataset=dataset#0为ml，1为pin
    global_epoch = 200
    num_users=0
    num_items=0
    
    # Load Data
    if global_dataset==0:
        #-----------------------movielens數據集-------------------------------
        ml1m_dir = 'data/ml-1m/ratings.dat'
        ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
        # Reindex
        user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
        user_id['userId'] = np.arange(len(user_id))
        ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
        item_id = ml1m_rating[['mid']].drop_duplicates()
        item_id['itemId'] = np.arange(len(item_id))
        ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
        ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
        print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
        print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
        sample_generator = SampleGenerator(ratings=ml1m_rating)
        num_users=6040
        num_items=3706
        #-----------------------------------------------------------------------
    else:
        #---------------------pinterest數據集------------------------------------
        # ========== Pinterest 数据加载 ==========
        pin_dir='data/pinterest/pinterest-20.train.rating'
        pin_rating=pd.read_csv(pin_dir,sep='\t',header=None,names=['uid','mid','rating','timestamp'])
        #Reindex
        user_id=pin_rating[['uid']].drop_duplicates().reindex()
        user_id['userId']=np.arange(len(user_id))
        pin_rating = pd.merge(pin_rating, user_id, on=['uid'], how='left')
        item_id = pin_rating[['mid']].drop_duplicates()
        item_id['itemId'] = np.arange(len(item_id))
        pin_rating = pd.merge(pin_rating, item_id, on=['mid'], how='left')
        pin_rating = pin_rating[['userId', 'itemId', 'rating', 'timestamp']]
        print('Range of userId is [{}, {}]'.format(pin_rating.userId.min(), pin_rating.userId.max()))
        print('Range of itemId is [{}, {}]'.format(pin_rating.itemId.min(), pin_rating.itemId.max()))
        sample_generator = SampleGenerator(ratings=pin_rating)
        num_users = pin_rating.userId.max() + 1
        num_items = pin_rating.itemId.max() + 1
        #------------------------------------------------------------------------
    data_tag='ml' if global_dataset==0 else 'pin'
    #model config
    gmf_config = {'alias': f'{data_tag}_gmf_factor{global_latent_dim}neg4',
                'num_epoch': global_epoch,#原200
                'batch_size': 1024,
                # 'optimizer': 'sgd',
                # 'sgd_lr': 1e-3,
                # 'sgd_momentum': 0.9,
                # 'optimizer': 'rmsprop',
                # 'rmsprop_lr': 1e-3,
                # 'rmsprop_alpha': 0.99,
                # 'rmsprop_momentum': 0,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users':num_users,
                'num_items':num_items,
                'latent_dim': global_latent_dim,
                'num_negative': 4,
                'l2_regularization': 0,  # 0.01
                'weight_init_gaussian': True,
                'use_cuda': True,
                'use_bachify_eval': True,#原False
                'device_id': 0,
                'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

    mlp_config = {'alias': f'{data_tag}_mlp_factor{global_latent_dim}neg4',
                'num_epoch': global_epoch,
                'batch_size': 256,  # 256
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users':num_users,
                'num_items':num_items,
                'latent_dim': global_latent_dim,
                'num_negative': 4,
                'layers': [global_latent_dim*2, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
                'weight_init_gaussian': True,
                'use_cuda': True,
                'use_bachify_eval': True,#原False
                'device_id': 0,
                'pretrain': False,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

    neumf_config = {'alias': f'{data_tag}_pretrained_neumf_factor{global_latent_dim}neg4',
                    'num_epoch': global_epoch,
                    'batch_size': 1024,
                    'optimizer': 'adam',
                    'adam_lr': 1e-3,
                    'num_users':num_users,
                    'num_items':num_items,
                    'latent_dim_mf': global_latent_dim,
                    'latent_dim_mlp': global_latent_dim,
                    'num_negative': 4,
                    'layers': [global_latent_dim*2, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
                    'l2_regularization': 0.0000001,
                    'weight_init_gaussian': True,
                    'use_cuda': True,
                    'use_bachify_eval': True,
                    'device_id': 0,
                    #neumf预训练信息修改 选择训练后的mf、mlp的最优hr轮次作为预训练模型
                    #ml:
                    #dim=8:ml_gmf_factor8neg4_Epoch21_HR0.6402_NDCG0.3646.model ml_mlp_factor8neg4_Epoch32_HR0.6575_NDCG0.3795.model
                    #dim=16:ml_gmf_factor16neg4_Epoch38_HR0.6912_NDCG0.4119.model ml_mlp_factor16neg4_Epoch12_HR0.6733_NDCG0.3988.model
                    #dim=32:ml_gmf_factor32neg4_Epoch20_HR0.7068_NDCG0.4314.model ml_mlp_factor32neg4_Epoch26_HR0.6945_NDCG0.4188.model
                    #dim=64:ml_gmf_factor64neg4_Epoch8_HR0.7076_NDCG0.4321.model ml_mlp_factor64neg4_Epoch22_HR0.6964_NDCG0.4196.model
                    #pin:
                    #dim=8:pin_gmf_factor8neg4_Epoch42_HR0.8696_NDCG0.5459.model pin_mlp_factor8neg4_Epoch26_HR0.8609_NDCG0.5353.model
                    #dim=16:pin_gmf_factor16neg4_Epoch5_HR0.8623_NDCG0.5355.model pin_mlp_factor16neg4_Epoch16_HR0.8647_NDCG0.5406.model
                    #dim=32:pin_gmf_factor32neg4_Epoch3_HR0.8568_NDCG0.5336.model pin_mlp_factor32neg4_Epoch12_HR0.8644_NDCG0.5399.model
                    #dim=64:pin_gmf_factor64neg4_Epoch2_HR0.8553_NDCG0.5332.model pin_mlp_factor64neg4_Epoch12_HR0.8664_NDCG0.5419.model
                    'pretrain': True,
                    'pretrain_mf': 'checkpoints/{}'.format('pin_gmf_factor64neg4_Epoch2_HR0.8553_NDCG0.5332.model'),
                    'pretrain_mlp': 'checkpoints/{}'.format('pin_mlp_factor64neg4_Epoch12_HR0.8664_NDCG0.5419.model'),
                    'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                    }

    # DataLoader for training
    evaluate_data = sample_generator.evaluate_data
    # Specify the exact model
    if global_model==0:
        config = gmf_config
        engine = GMFEngine(config)
    elif global_model==1:
        config = mlp_config
        engine = MLPEngine(config)
    elif global_model==2:
        config = neumf_config
        engine = NeuMFEngine(config)
    else:
        raise ValueError(f'Unknown model type: {global_model}')
    #早停配置，自動判斷收斂與否
    PATIENCE = 8          # 连续PATIENCE个epoch最优HR没提升就停
    MIN_DELTA = 1e-4       # 提升小于这个值也算没提升（防抖动）
    best_hr = 0.0
    best_epoch = -1
    no_improve = 0
    for epoch in range(config['num_epoch']):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
        # ---- 判断是否提升 ----
        if hit_ratio > best_hr + MIN_DELTA:
            best_hr = hit_ratio
            best_epoch = epoch
            no_improve = 0
            #只在出现最优HR时保存checkpoint
            engine.save(config['alias'], epoch, hit_ratio, ndcg)
            print('>>> New best HR = {:.4f} at epoch {}'.format(best_hr, best_epoch))
        else:
            no_improve += 1
            print('>>> No improvement for {} epoch(s), best HR = {:.4f} (epoch {})'
                .format(no_improve, best_hr, best_epoch))

        # ---- 触发早停 ----epoch>20时才能触发
        if no_improve >= PATIENCE and  epoch>20:
            print('=' * 80)
            print('Early stopping at epoch {}. Best HR = {:.4f} at epoch {}'
                .format(epoch, best_hr, best_epoch))
            print('=' * 80)
            break

    print('Training finished. Best HR = {:.4f} at epoch {}'.format(best_hr, best_epoch))
    return 
