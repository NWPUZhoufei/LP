import argparse

def parse_args_batch_train():
    parser = argparse.ArgumentParser(description='batch_train')
    parser.add_argument('--dataset', default='miniImageNet', help='training base model')  
    parser.add_argument('--data_path', default='./datasets/miniImagenet', help='train data path')
    parser.add_argument('--image_size'  , default=224, type=int,  help='image size')
    parser.add_argument('--base_class' , default=64, type=int, help='total number of classes in base class') 
    parser.add_argument('--batch_size' , default=16, type=int, help='total number of batch size in base class')
    parser.add_argument('--feature_size' , default=512, type=int, help='feature_size')
    parser.add_argument('--backbone', default='ResNet10', help='backbone type')
    parser.add_argument('--list_of_out_dims', default=[64,128,256,512], help='every block output')
    parser.add_argument('--list_of_stride', default=[1,2,2,2], help='every block conv stride')
    parser.add_argument('--list_of_dilated_rate', default=[1,1,1,1], help='dilated conv')  
    parser.add_argument('--method', default='Linear_Classifier', help='Linear_Classifier/ProtoNet') 
    parser.add_argument('--train_aug', default='True',  help='perform data augmentation or not during training ') 
    parser.add_argument('--save_freq', default=100, type=int, help='Save frequency')
    parser.add_argument('--save_dir', default='./logs', help='Save dir')
    parser.add_argument('--epoch', default=400, type=int, help ='total batch train epoch')  
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
    parser.add_argument('--seed' , default=1111, type=int, help='feature_size')
    parser.add_argument('--pretrain_model_path', default='./log/best_model.tar', help='model_path')
    return parser.parse_args()

def parse_args_eposide_train():
    parser = argparse.ArgumentParser(description='eposide_train')
    parser.add_argument('--dataset', default='miniImageNet', help='training base model')  
    parser.add_argument('--source_data_path', default='./datasets/miniImagenet', help='train data path')  
    parser.add_argument('--image_size'  , default=224, type=int,  help='image size')  
    parser.add_argument('--base_class' , default=64, type=int, help='total number of classes in in base class')  
    parser.add_argument('--backbone', default='ResNet10', help='backbone type')
    parser.add_argument('--list_of_out_dims', default=[64,128,256,512], help='every block output')
    parser.add_argument('--list_of_stride', default=[1,2,2,2], help='every block conv stride')
    parser.add_argument('--list_of_dilated_rate', default=[1,1,1,1], help='dilated conv')  
    parser.add_argument('--method', default='ProtoNet', help='Linear_Classifier/ProtoNet') 
    parser.add_argument('--train_aug', default='True',  help='perform data augmentation or not during training ') 
    parser.add_argument('--save_freq', default=50, type=int, help='Save frequency')
    parser.add_argument('--save_dir', default='./logs', help='Save dir')
    parser.add_argument('--epoch', default=50, type=int, help ='total batch train epoch')  
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
    parser.add_argument('--n_way', default=5, type=int,  help='class num to classify for every task')
    parser.add_argument('--n_support', default=5, type=int,  help='number of labeled data in each class, same as n_support') 
    parser.add_argument('--n_query', default=15, type=int,  help='number of test data in each class, same as n_query') 
    parser.add_argument('--n_eposide', default=100, type=int, help ='total task every epoch') # for meta-learning methods, each epoch contains 100 episodes
    parser.add_argument('--pretrain_model_path', default='./log/best_model.tar', help='model_path')
    parser.add_argument('--seed' , default=1111, type=int, help='feature_size')
    parser.add_argument('--crop_size' , default=96, type=int, help='crop_size')
    parser.add_argument('--crop_num' , default=6, type=int, help='crop_num')
    parser.add_argument('--min_scale_crops', default=0.05, type=float, help='min_scale_crops')
    parser.add_argument('--max_scale_crops', default=0.14, type=float, help='max_scale_crops')
    parser.add_argument('--lamba_diversity_loss', default=0.3, type=float, help='variance_loss')
    parser.add_argument('--variance_loss_beta', default=5e-04, type=float, help='variance_loss_beta')
    parser.add_argument('--test_n_eposide', default=600, type=int, help ='total task every epoch') # for meta-learning methods, each epoch contains 100 episodes
    return parser.parse_args()

def parse_args_test():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--image_size'  , default=224, type=int,  help='image size') 
    parser.add_argument('--feature_size' , default=512, type=int, help='feature_size')
    parser.add_argument('--list_of_out_dims', default=[64,128,256,512], help='every block output')
    parser.add_argument('--list_of_stride', default=[1,2,2,2], help='every block conv stride')
    parser.add_argument('--list_of_dilated_rate', default=[1,1,1,1], help='dilated conv') 
    parser.add_argument('--model_path', default='./log/best_model.tar', help='model_path')
    parser.add_argument('--n_way', default=5, type=int,  help='class num to classify for every task')
    parser.add_argument('--n_support', default=5, type=int,  help='number of labeled data in each class, same as n_support') 
    parser.add_argument('--n_query', default=15, type=int,  help='number of test data in each class, same as n_query') 
    parser.add_argument('--test_n_eposide', default=600, type=int, help ='total task every epoch') # for meta-learning methods, each epoch contains 100 episodes
    parser.add_argument('--seed' , default=1111, type=int, help='feature_size')
    parser.add_argument('--current_data_path', default='./datasets/ISIC', help='ISIC_data_path') 
    parser.add_argument('--current_class', default=7, type=int, help='total number of classes in ISIC')
    parser.add_argument('--patch_size' , default=112, type=int, help='patch_size')
    parser.add_argument('--tr_N' , default=7, type=int, help='tr_N')
    parser.add_argument('--tr_K' , default=10, type=int, help='tr_K')
    return parser.parse_args()










