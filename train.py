import train_dataset # train
import torch
import os
import numpy as np
from io_utils import parse_args_eposide_train
import ResNet10
import torch.nn as nn
import random
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings("ignore", category=Warning)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

# X （n，c） Y （m，c）
# pred （m，n）
def global_matching(X, Y):
    n, c = X.size()
    m, _ = Y.size()
    X = X.unsqueeze(0).expand(m, n, c) # (m,n,c)
    Y = Y.unsqueeze(1).expand(m, n, c) # (m,n,c)
    dist = torch.pow(X - Y, 2).sum(2) # (m,n)
    pred = -dist
    return pred


# X （way*query, way，patch, c） Y （way*query,patch,c）
# pred （way*query,way,patch）
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

        
        
def train(train_loader, model, loss_fn, optimizer, params):
    model.train()
    total_loss = 0
    softmax = torch.nn.Softmax(dim=2)
    relu = nn.ReLU()
    for i, x in enumerate(train_loader):
        optimizer.zero_grad() 
        x_crop = torch.stack(x[0:params.crop_num]).cuda() # (crop_num,way,shot+query,3,crop_size,crop_size)
        x_raw = x[params.crop_num].cuda() # (way,shot+query,3,224,224)
        
        # local
        x_crop = x_crop.contiguous().view(params.crop_num*params.n_way*(params.n_support+params.n_query), 3, params.crop_size, params.crop_size)# (6*way*query,3,96,96)
        x_crop = model(x_crop) # (6*way*(shot+query),512,7,7)
        x_crop = torch.mean(x_crop, [2,3]) # (6*way*(shot+query),512)
        x_crop = x_crop.view(params.crop_num, params.n_way, params.n_support+params.n_query, 512)
        # variance loss        
        std = torch.sqrt(x_crop.var(dim=0) + params.variance_loss_beta) # (-,，-，512)
        variance_loss = torch.mean(relu(1 - std))
        
        support_set_local = x_crop[:,:,:params.n_support,:] #(crop_num, way, shot, 512)
        support_set_local = support_set_local.permute(1,0,2,3) # (way,crop_num,shot,512) 
        support_set_local = support_set_local.permute(0,2,1,3) # (way,shot,crop_num,512 ) 
        
        query_set_local = x_crop[:,:,params.n_support:,:]  
        query_set_local = query_set_local.contiguous().view(params.crop_num, params.n_way*params.n_query, 512) # (6, 5*15, 512)
        query_set_local = query_set_local.permute(1,0,2) 
        
        # global
        x_raw = x_raw.contiguous().view(params.n_way*(params.n_support+params.n_query),3,224,224) # (way*(shot+query),3,224,224)
        x_raw = model(x_raw) # (way*(shot+query),512，7，7)
        x_raw = x_raw.view(params.n_way, params.n_support+params.n_query, 512, 7, 7) 
        support_set_raw = x_raw[:,:params.n_support,:, :, :]  # (way,shot,512,7,7)
        support_set_prototype = torch.mean(support_set_raw, [1,3,4]) # (way,512)
        query_set_raw = x_raw[:,params.n_support:,:, :, :] # (way,query,512)
        query_set_raw = torch.mean(query_set_raw, [3,4])
        query_set_raw = query_set_raw.contiguous().view(params.n_way*params.n_query, 512) # (way*query,512) 
        global_pred = global_matching(support_set_prototype, query_set_raw)
        
        
        local_prototypes = prompt_search(query_set_local, support_set_local)
        
        # local pred
        local_pred = local_matching(local_prototypes, query_set_local)
        local_pred_weight = softmax(local_pred) #（way*query，way, patch）
        local_pred = local_pred * local_pred_weight
        local_pred = torch.sum(local_pred, 2) # (way*query,way)
        
        
        
       
        # loss
        pred = global_pred + local_pred 
            
        # ce loss
        query_set_y = torch.from_numpy(np.repeat(range(params.n_way), params.n_query))
        query_set_y = query_set_y.cuda()
        loss_ce = loss_fn(pred, query_set_y) 
        
        loss = loss_ce + variance_loss * params.lamba_diversity_loss
        
        
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()
    avg_loss = total_loss/float(i+1)
    return avg_loss
        
 
                
if __name__=='__main__':

    params = parse_args_eposide_train()

    setup_seed(params.seed)

    datamgr_train = train_dataset.Eposide_DataManager(crop_size=params.crop_size, crop_num=params.crop_num, min_scale_crops=params.min_scale_crops, max_scale_crops=params.max_scale_crops, data_path=params.source_data_path, num_class=params.base_class, n_way=params.n_way, n_support=params.n_support, n_query=params.n_query, n_eposide=params.n_eposide)
    train_loader = datamgr_train.get_data_loader()

    model = ResNet10.ResNet(list_of_out_dims=params.list_of_out_dims, list_of_stride=params.list_of_stride, list_of_dilated_rate=params.list_of_dilated_rate)

    if not os.path.isdir(params.save_dir):
        os.makedirs(params.save_dir)

    tmp = torch.load(params.pretrain_model_path)
    state = tmp['state']
    model.load_state_dict(state)
    model = model.cuda()


    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam([{"params":model.parameters()}], lr=params.lr)

    for epoch in range(params.epoch):
        train_loss = train(train_loader, model, loss_fn, optimizer, params)
        print('train:', epoch+1, 'current epoch train loss:', train_loss)
        if (epoch+1) % params.save_freq==0:
            outfile = os.path.join(params.save_dir, '{:d}.tar'.format(epoch+1))
            torch.save({'epoch':epoch+1, 'state_model':model.state_dict()}, outfile) 



    
    
    
    
    