from os.path import  basename,join
import torch
from numpy import sum

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

data_path_base = './data/'
data_path_res = './data/restaurant/'
data_path_out='./data/outdoor/'
data_path_class='./data/classroom/'
gt_path = './data/original/ground_truth_csv/'
model_path = './final_models/mcnn_shtechA_660.h5'

output_dir_base = './output/'
model_name =basename(model_path).split('.')[0]
#file_results = os.path.join(output_dir, 'results_' + model_name + '_.txt')
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)
# output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)

net = CrowdCounter()

trained_model = join(model_path)
network.load_net(trained_model, net)
net.cuda()
net.eval()

# load test data
no_gt = True#True:not need ground_truth,for just estimate the crowd number;False:need ground_truth,need to calculate mae,mse
data_loader_res = ImageDataLoader(data_path_res, gt_path, shuffle=False, gt_downsample=True, pre_load=True, no_gt=True)
data_loader_out = ImageDataLoader(data_path_out, gt_path, shuffle=False, gt_downsample=True, pre_load=True, no_gt=True)
data_loader_class = ImageDataLoader(data_path_class, gt_path, shuffle=False, gt_downsample=True, pre_load=True, no_gt=True)

# for blob in data_loader:
#     im_data = blob['data']
#     if no_gt==False:
#         gt_data = blob['gt_density']
#     density_map = net(im_data)
#     density_map = density_map.data.cpu().numpy()
#     et_count = np.sum(density_map)#predicted_count
#     if save_output:
#         utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')
#     print('et_count:'+str(et_count))

#重新加载数据
def reloadData():
    data_loader_res = ImageDataLoader(data_path_res, gt_path, shuffle=False, gt_downsample=True, pre_load=True,
                                      no_gt=True)
    data_loader_out = ImageDataLoader(data_path_out, gt_path, shuffle=False, gt_downsample=True, pre_load=True,
                                      no_gt=True)
    data_loader_class = ImageDataLoader(data_path_class, gt_path, shuffle=False, gt_downsample=True, pre_load=True,
                                        no_gt=True)

#返回指定下标的图像的人数
def returnNum(category,index):
    index=index-1
    if category=='restaurant':
        blob=data_loader_res.blob_list[index]
    elif category=='outdoor':
        blob = data_loader_out.blob_list[index]
    elif category=='classroom':
        blob = data_loader_class.blob_list[index]
    im_data=blob['data']
    density_map = net(im_data)
    density_map = density_map.data.cpu().numpy()
    et_count = sum(density_map)  # preedicted_count
    if save_output:
        output_dir=output_dir_base+category+'/'
        utils.save_density_map(density_map, output_dir, category+'_' + blob['fname'].split('.')[0] + '.png')
    return et_count

# if __name__=="__main__":
#     data_loader = reloadDate()
#     tmp = data_loader.blob_list[0]['data']
#     density_map = net(tmp)
#     density_map = density_map.data.cpu().numpy()
