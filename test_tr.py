import numpy as np
from io_utils import parse_args_test
import test_dataset
import ResNet10
import torch
import random
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings("ignore", category=Warning)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def global_matching(X, Y):
    n, c = X.size()
    m, _ = Y.size()
    X = X.unsqueeze(0).expand(m, n, c) # (m,n,c)
    Y = Y.unsqueeze(1).expand(m, n, c) # (m,n,c)
    dist = torch.pow(X - Y, 2).sum(2) # (m,n)
    pred = -dist
    return pred

def local_matching(X, Y):
    Y = Y.unsqueeze(1).expand(Y.shape[0], X.shape[1], Y.shape[1], Y.shape[2]) # (way*query,way,patch,c)
    dist = torch.pow(X - Y, 2).sum(3) # (way*query,way,patch)
    pred = -dist  # (way*query,way,patch)     
    return pred

def prompt_search(query_set_patch, support_set_patch): 
    way_query, num_patch, c = query_set_patch.size()
    way, shot, _, _  = support_set_patch.size()
    local_prototype = torch.rand(way_query, way, shot, num_patch, 512).cuda()
    for i in range(way_query):
        current_query_set_patch = query_set_patch[i] # (patch,512)
        for j in range(way):
            for k in range(shot):
                current_support_image_patch = support_set_patch[j][k] # (patch,512)
                cost = -global_matching(current_support_image_patch, current_query_set_patch) # (query_patch, support_patch)
                cost = cost.cpu().detach().numpy()
                row_ind, col_ind = linear_sum_assignment(cost)
                for l in range(num_patch):
                    local_prototype[i][j][k][l] = current_support_image_patch[col_ind[l]]
    local_prototype = torch.mean(local_prototype, 2) # (way*query,way,patch,512)         
    return local_prototype

def prompt_search_tr(query_set_patch, support_set_patch): 
    way_query, num_patch, c = query_set_patch.size()
    way = len(support_set_patch)
    local_prototype = torch.rand(way_query, way, num_patch, 512).cuda()
    for i in range(way_query):
        current_query_set_patch = query_set_patch[i] # (patch,512)
        for j in range(way):
            shot = support_set_patch[j].size(0) # tensor（shot+k，512）
            current_local_prototype = torch.rand(shot, num_patch, 512).cuda()
            for k in range(shot):
                current_support_image_patch = support_set_patch[j][k] # (patch,512)
                cost = -global_matching(current_support_image_patch, current_query_set_patch) # (query_patch, support_patch)
                cost = cost.cpu().detach().numpy()
                row_ind, col_ind = linear_sum_assignment(cost)
                for l in range(num_patch):
                    current_local_prototype[k][l] = current_support_image_patch[col_ind[l]]
            current_local_prototype = torch.mean(current_local_prototype, 0) # (num_patch,512)
            local_prototype[i][j] = current_local_prototype # （patch，512）
    # (way*query,way,patch,512)         
    return local_prototype

# support_set_raw (-,512)
# support_set_raw_gt (-)
# query_set_raw (75,512)   
# pred (75,5)
def get_global_pred(support_set_raw, support_set_raw_gt, query_set_raw, softmax1):
    out_support = support_set_raw.cpu().numpy() # (5*5,512)
    out_query = query_set_raw.cpu().numpy() # (75,512)
    classifier = LogisticRegression(max_iter=1000).fit(X=out_support, y=support_set_raw_gt)
    global_pred_LR = classifier.predict_proba(out_query)
    global_pred_LR = torch.from_numpy(global_pred_LR).cuda() # (75,5)
    global_pred_LR = softmax1(global_pred_LR)
    return global_pred_LR

def get_local_pred(query_set_patch, local_prototypes, softmax2, params):
    num_patches = int((params.image_size/params.patch_size)**2) 
    local_prototypes = local_prototypes.cpu().numpy() # (75,5,6,512)
    local_query = query_set_patch.cpu().numpy() # (75,6,512)
    # local pred
    local_pred_LR = []
    for i in range(params.n_way*params.n_query):
        for j in range(num_patches):
            current_local_query_feature = local_query[i][j] # (512)
            current_local_query_feature = np.expand_dims(current_local_query_feature, 0) # (1,512)
            current_local_prototype = local_prototypes[i, :, j] # (5,512)
            y = np.tile(range(params.n_way), 1)
            y.sort()
            classifier = LogisticRegression(max_iter=1000).fit(X=current_local_prototype, y=y)
            current_local_feature_pred = classifier.predict_proba(current_local_query_feature) # (1,5)
            current_local_feature_pred = torch.from_numpy(current_local_feature_pred) # (1,5)
            local_pred_LR.append(current_local_feature_pred)  
    local_pred_LR = torch.stack(local_pred_LR).squeeze(1) # (75*6,5)
    local_pred_LR = local_pred_LR.view(params.n_way*params.n_query, num_patches, params.n_way).cuda()
    local_pred_LR = softmax2(local_pred_LR)
    return local_pred_LR

def global_local_mean(global_pred_LR, local_pred_LR, softmax1, params):
    local_pred_LR_weight = softmax1(local_pred_LR) #（way*query，way, patch）
    current_pred = local_pred_LR * local_pred_LR_weight
    local_pred_LR_mean = torch.sum(current_pred, 1) # (way*query,way)
    local_pred_LR_mean = softmax1(local_pred_LR_mean)
    pred = global_pred_LR + local_pred_LR_mean 
    return pred

def get_local_feature(x, model, params):
    # get patch
    num_patches = int((params.image_size/params.patch_size)**2) 
    x_patch = nn.Unfold(kernel_size=params.patch_size, stride=params.patch_size)(x) # (B,patch_size*patch_size*channel,patch_num*patch_num)
    x_patch = x_patch.transpose(2,1)
    x_patch = x_patch.view(params.n_way*(params.n_support+params.n_query), num_patches, 3, params.patch_size, params.patch_size)
    # (way*(shot+query), num_patch, 3, patch size, patch size)
    x_patch = x_patch.contiguous().view(params.n_way*(params.n_support+params.n_query)*num_patches, 3, params.patch_size, params.patch_size)
    x_patch = model(x_patch) # (params.n_way*(params.n_support+params.n_query)*num_patch, 512, 4, 4)
    x_patch = torch.mean(x_patch, [2,3]) # (params.n_way*(params.n_support+params.n_query)*num_patch, 512)
    beta = 0.5
    x_patch = torch.pow(x_patch, beta) 
    x_patch = x_patch.view(params.n_way, params.n_support+params.n_query, num_patches, 512)
    support_set_patch = x_patch[:, :params.n_support, :, :] # (way,shot,num_patch,512)
    query_set_patch = x_patch[:, params.n_support:, :, :] # (way,qeury,num_patch,512)
    #support_set_patch = support_set_patch.contiguous().view(params.n_way, params.n_support*num_patches, 512)
    query_set_patch = query_set_patch.contiguous().view(params.n_way*params.n_query, num_patches, 512)
    return support_set_patch, query_set_patch

def get_global_feature(x, model, params):
    x_raw = model(x) # (way*(shot+query),512,7,7)
    x_raw = torch.mean(x_raw, [2,3])
    beta = 0.5
    x_raw = torch.pow(x_raw, beta) 
    x_raw = x_raw.view(params.n_way, params.n_support+params.n_query, 512)
    support_set_raw = x_raw[:, :params.n_support, :]  # (way,shot,512)
    query_set_raw = x_raw[:,params.n_support:,:] # (way,query,512)
    query_set_raw = query_set_raw.contiguous().view(params.n_way*params.n_query, 512) 
    return support_set_raw, query_set_raw

    

def Tr_global_1shot(support_set, query_set, pred, K):
        # pred （75，5）
        softmax = torch.nn.Softmax(dim=1)
        pred = softmax(pred)
        score, index = pred.max(1)
        all_class_score = []
        all_class_index = []
        for j in range(5):
            current_class_score = []
            current_class_index = []
            for i in range(75):
                if index[i]==j:
                    current_class_score.append(score[i])
                    current_class_index.append(i)
            all_class_score.append(current_class_score)
            all_class_index.append(current_class_index)
        
        prototype = []
        for i in range(5):
            current_class_index = all_class_index[i]
            if len(current_class_index) == 0:
                current_prototype = torch.mean(support_set[i],0) # (640)
            elif len(current_class_index) <= K:
                current_query_image = query_set[current_class_index] # (1,640)  
                current_prototype = torch.cat((support_set[i], current_query_image),0) # (shot+1,640)
                current_prototype = torch.mean(current_prototype, 0) # (640)
            else:
                current_class_score = all_class_score[i]
                current_class_score_index = np.argsort(current_class_score)
                current_class_index = np.array(current_class_index)[current_class_score_index[-K:].tolist()] 
                current_query_image = query_set[current_class_index] # (2,640) n>=2
                current_prototype = torch.cat((support_set[i], current_query_image),0) # (shot+2,640)
                current_prototype = torch.mean(current_prototype, 0) # (640)
            prototype.append(current_prototype)
            
        prototype = torch.stack(prototype, 0) #(5,640)
        support_set_raw_tr_gt = np.tile(range(5), 1)
        support_set_raw_tr_gt.sort()
        return prototype, support_set_raw_tr_gt

def Tr_global_5shot(support_set, query_set, pred, K):
        # pred （75，5）
        softmax = torch.nn.Softmax(dim=1)
        pred = softmax(pred)
        score, index = pred.max(1)
        all_class_score = []
        all_class_index = []
        for j in range(5):
            current_class_score = []
            current_class_index = []
            for i in range(75):
                if index[i]==j:
                    current_class_score.append(score[i])
                    current_class_index.append(i)
            all_class_score.append(current_class_score)
            all_class_index.append(current_class_index)
        
        tr_support_set = []
        for i in range(5):
            current_class_index = all_class_index[i]
            if len(current_class_index) == 0:
                current_support_set = support_set[i] # (shot,640)
            elif len(current_class_index) <= K:
                current_query_image = query_set[current_class_index] # (1,640)  
                current_support_set = torch.cat((support_set[i], current_query_image),0) # (shot+k,640)
            else:
                current_class_score = all_class_score[i]
                current_class_score_index = np.argsort(current_class_score)
                current_class_index = np.array(current_class_index)[current_class_score_index[-K:].tolist()]
                current_query_image = query_set[current_class_index] # (2,640) n>=2
                current_support_set = torch.cat((support_set[i], current_query_image),0) # (shot+K,640)
            
            tr_support_set.append(current_support_set)
        tr_support_set_all = torch.cat((tr_support_set[0], tr_support_set[1], tr_support_set[2], tr_support_set[3], tr_support_set[4]),0) # (-,640)
        tr_support_set_gt = [0] * len(tr_support_set[0]) + [1] * len(tr_support_set[1]) + [2] * len(tr_support_set[2]) + [3] * len(tr_support_set[3]) + [4] * len(tr_support_set[4])
        tr_support_set_gt = np.array(tr_support_set_gt)
        
        return tr_support_set_all, tr_support_set_gt
    

def Tr_local(support_set_patch, query_set_patch, pred, K):
    softmax = torch.nn.Softmax(dim=1)
    pred = softmax(pred)
    score, index = pred.max(1)
    all_class_score = []
    all_class_index = []
    for j in range(5):
        current_class_score = []
        current_class_index = []
        for i in range(75):
            if index[i]==j:
                current_class_score.append(score[i])
                current_class_index.append(i)
        all_class_score.append(current_class_score)
        all_class_index.append(current_class_index)
    
    support_set_patch_tr = []
    for i in range(5):
        current_class_index = all_class_index[i]
        if len(current_class_index) == 0:
            current_support_set = support_set_patch[i] # (shot,num_patch,512)
        elif len(current_class_index) <= K:
            current_query_image = query_set_patch[current_class_index] # (k,num_patch,512)
            current_support_set = torch.cat((support_set_patch[i], current_query_image),0) # (shot+k,num_patch,512)
        else:
            current_class_score = all_class_score[i]
            current_class_score_index = np.argsort(current_class_score)
            current_class_index = np.array(current_class_index)[current_class_score_index[-K:].tolist()] 
            current_query_image = query_set_patch[current_class_index] # (k,num_patch,512) n>=2
            current_support_set = torch.cat((support_set_patch[i], current_query_image),0) # (shot+K,num_patch,512)
        support_set_patch_tr.append(current_support_set)
    return support_set_patch_tr
    


def test(novel_loader, model, params):
    iter_num = len(novel_loader) 
    acc = []
    softmax1 = torch.nn.Softmax(dim=1)
    softmax2 = torch.nn.Softmax(dim=2)
    with torch.no_grad():
        for i, (x,_) in enumerate(novel_loader):
            x = x.contiguous().view(params.n_way*(params.n_support+params.n_query), *x.size()[2:]).cuda()
            support_set_patch, query_set_patch = get_local_feature(x, model, params)
            support_set_raw, query_set_raw = get_global_feature(x, model, params)   
            support_set_raw_gt = np.tile(range(params.n_way), params.n_support)
            support_set_raw_gt.sort()
            global_pred_LR = get_global_pred(support_set_raw.contiguous().view(params.n_way*params.n_support,512), support_set_raw_gt, query_set_raw, softmax1)
            local_prototypes = prompt_search(query_set_patch, support_set_patch)
            local_pred_LR = get_local_pred(query_set_patch, local_prototypes, softmax2, params)

            pred = global_local_mean(global_pred_LR, local_pred_LR, softmax1, params)

            
            for i in range(params.tr_N):
                if params.n_support==1:
                    support_set_raw_tr, support_set_raw_tr_gt = Tr_global_1shot(support_set_raw, query_set_raw, pred, params.tr_K)
                else:
                    support_set_raw_tr, support_set_raw_tr_gt = Tr_global_5shot(support_set_raw, query_set_raw, pred, params.tr_K)

                global_pred_tr = get_global_pred(support_set_raw_tr, support_set_raw_tr_gt, query_set_raw, softmax1)
                support_set_patch_tr = Tr_local(support_set_patch, query_set_patch, pred, params.tr_K)
                local_prototypes_tr = prompt_search_tr(query_set_patch, support_set_patch_tr)
                local_pred_tr = get_local_pred(query_set_patch, local_prototypes_tr, softmax2, params)
                pred = global_local_mean(global_pred_tr, local_pred_tr, softmax1, params)
                
            y_query = np.repeat(range(params.n_way), params.n_query)
            topk_scores, topk_labels = pred.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:,0] == y_query)
            correct_this, count_this = float(top1_correct), len(y_query)
            acc.append((correct_this/ count_this *100))
            
    acc_all  = np.asarray(acc)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all) 
    print('acc : %4.2f%% +- %4.2f%%' %(acc_mean, 1.96* acc_std/np.sqrt(iter_num)))   


if __name__=='__main__':
    
    params = parse_args_test()
    setup_seed(params.seed)
    
    datamgr = test_dataset.Eposide_DataManager(data_path=params.current_data_path, num_class=params.current_class, image_size=params.image_size, n_way=params.n_way, n_support=params.n_support, n_query=params.n_query, n_eposide=params.test_n_eposide)
    novel_loader = datamgr.get_data_loader(aug=False) 
    
    model = ResNet10.ResNet(list_of_out_dims=params.list_of_out_dims, list_of_stride=params.list_of_stride, list_of_dilated_rate=params.list_of_dilated_rate)
    tmp = torch.load(params.model_path)
    state_model = tmp['state_model']
    model.load_state_dict(state_model)
    model.cuda()
    model.eval()

    test(novel_loader, model, params)
   
    