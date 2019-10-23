import numpy as np
import sys, os
import time
import cv2
sys.path.append(os.getcwd())
# crnn packages
import torch
from torch.autograd import Variable
import utils
import models.crnn as crnn
import alphabets
import params
import pickle as pkl
import argparse
from dataset_v2 import baiduDataset
from torch.utils.data import DataLoader
import torch
import utils



date_model = '/home/song/workplace/OCR/crnn_chinese_characters_rec/date_adam_bak/crnn_Rec_done_174_0.9490291262135923.pth'
birth_d_model = '/home/song/workplace/OCR/crnn_chinese_characters_rec/birth_d/crnn_Rec_done_51_1.002540650406504.pth'
birth_y_model = '/home/song/workplace/OCR/crnn_chinese_characters_rec/birth_y/crnn_Rec_done_7_0.9929435483870968.pth'
birth_m_model = "/home/song/workplace/OCR/crnn_chinese_characters_rec/birth_d/crnn_Rec_done_51_1.002540650406504.pth"
name_model = '/home/song/workplace/OCR/crnn_chinese_characters_rec/name/crnn_Rec_done_65_0.9944556451612904.pth'
gender_model = '/home/song/workplace/OCR/crnn_chinese_characters_rec/gender/crnn_Rec_done_3_0.9939516129032258.pth'
ethnic_model = '/home/song/workplace/OCR/crnn_chinese_characters_rec/ethnic/crnn_Rec_done_37_0.9909274193548387.pth'
id_model = '/home/song/workplace/OCR/crnn_chinese_characters_rec/id/crnn_Rec_done_10_0.9879032258064516.pth'
aa_model = '/home/song/workplace/OCR/crnn_chinese_characters_rec/agency_addr/crnn_Rec_done_132_0.8388671875.pth'
parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, default='/uuz/song/datasets/OCR/test_xxx', help='the path to your images')
parser.add_argument('--model_path', type=str, default='/uuz/song/datasets/OCR/train_gen/train_part_vx/xxx', help='model path')
parser.add_argument('--save_path', type=str, default='/uuz/song/datasets/OCR/train_gen/train_part_vx/xxx', help='result path')
parser.add_argument('--item', type=str, default='xxx', help='attribute name')
opt = parser.parse_args()
# crnn params
# 3p6m_third_ac97p8.pth
# crnn_model_path = 'trained_models/crnn_Rec_done_1.pth'
# crnn_model_path = 'trained_models/mixed_second_finetune_acc97p7.pth'
crnn_model_path = aa_model
alphabet = alphabets.alphabet_agency_addr
nclass = len(alphabet)+1

# crnn文本信息识别
def crnn_recognition(cropped_image, model):

    converter = utils.strLabelConverter(alphabet)
    image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    ### ratio
    ### 280是中文训练集中图片的宽度，160是将原始图片缩小后的图片宽度
    # w_now = int(image.shape[1] / (280 * 1.0 / params.imgW))
    h, w = image.shape
    # image = cv2.resize(image, (0,0), fx=w_now/w, fy=params.imgH/h, interpolation=cv2.INTER_CUBIC)
    image = cv2.resize(image, (0,0), fx=params.imgW/w, fy=params.imgH/h, interpolation=cv2.INTER_CUBIC)
    image = (np.reshape(image, (params.imgH, params.imgW, 1))).transpose(2, 0, 1)

    # image = (np.reshape(image, (params.imgH, w_now, 1))).transpose(2, 0, 1)
    image = image.astype(np.float32) / 255.
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image.sub_(params.mean).div_(params.std)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # print('results: {0}'.format(sim_pred))
    return sim_pred


def pred(model_path, save_path, item_name, debug_file):
    if item_name not in ['name', 'birth_y']
    model = crnn.CRNN(32, 1, nclass, 256)   
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from {0}'.format(model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(model_path))
    
    started = time.time()
    ## read an image
    i = 0
    res_list = []
    for img in os.listdir(opt.images_path):
        img_path = os.path.join(opt.images_path, img)
        image = cv2.imread(img_path)

        pred = crnn_recognition(image, model)
        res_list.append({
            'file_name':img,
            item_name:pred
        })
        with open (debug_file, 'a') as f:
            f.writelines(img +" : "+pred+'\n')
        i = i + 1
        if(i % 200==0):
            print(pred)
            print('{} samples are completed'.format(i))

    with open(save_path, 'wb') as f:
        pkl.dump(res_list, f, protocol=2)
    finished = time.time()
    print('elapsed time: {0}'.format(finished-started))
    
  
def debug(model_path, debug_pkl_file, debug_txt_file, max_i=1000):
    val_dataset = baiduDataset("/uuz/song/datasets/OCR/train_gen/train_part_v3/train_items_part_v3_p1_aa", "/uuz/song/datasets/OCR/train_gen/train_part_v3/val_label_p1_{}.txt".format(params.experiment), params.alphabet, False, (params.imgW, params.imgH))
    val_loader = DataLoader(val_dataset, batch_size=params.val_batchSize, shuffle=False, num_workers=params.workers)
    converter = utils.strLabelConverter(val_dataset.alphabet)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = crnn.CRNN(32, 1, nclass, 256)   
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from {0}'.format(model_path))
    model.load_state_dict(torch.load(model_path))

    print('Start val')
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    i = 0
    n_correct = 0
    debug_list = []
    for i_batch, (image, index) in enumerate(val_loader):
        image = image.to(device)
        label = utils.get_batch_label(val_dataset, index)
        preds = model(image)
        preds = preds.to(torch.float64)
        preds = preds.to(device)
        batch_size = image.size(0)
        index = np.array(index.data.numpy())
        text, length = converter.encode(label)
        text = text.to(device)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, label):
            if pred == target:
                n_correct += 1
            else:
                debug_list.append({
                    'pred':pred,
                    'target':target
                })
                with open(debug_txt_file, 'a') as f:
                    f.write(pred + " " + target + "\n")

        # if i_batch == max_i:
        #     break
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, label):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    # print(n_correct)
    print("{} of {} is correct".format(n_correct, (i_batch+1) * params.val_batchSize))
    accuracy = n_correct / float((i_batch+1) * params.val_batchSize)
    # with open(debug_pkl_file, 'wb') as f:
    #     pkl.dump(debug_list, f)
    return accuracy


if __name__ == '__main__':
    # debug(model_path='/uuz/song/datasets/OCR/train_gen/train_part_v3/model/aa/crnn_best.pth',
    #     debug_pkl_file='/uuz/song/datasets/OCR/train_gen/train_part_v3/model/aa/debug.pkl',
    #     debug_txt_file='/uuz/song/datasets/OCR/train_gen/train_part_v3/model/aa/debug.txt')
    pred(model_path = '/uuz/song/datasets/OCR/train_gen/train_part_v3/model/aa/crnn_best.pth',
        save_path='/uuz/song/datasets/OCR/test_gen/test_part_v3/res_aa_p1.pkl',
        item_name='aa',
        debug_file = '/uuz/song/datasets/OCR/test_gen/test_part_v3/res_aa_p1.txt')

	# crnn network
    # model = crnn.CRNN(32, 1, nclass, 256)
    # if torch.cuda.is_available():
    #     model = model.cuda()
    # print('loading pretrained model from {0}'.format(crnn_model_path))
    # # 导入已经训练好的crnn模型
    # model.load_state_dict(torch.load(crnn_model_path))
    
    # started = time.time()
    # ## read an image
    # i = 0
    # res_list = []
    # for img in os.listdir(opt.images_path):
    #     img_path = os.path.join(opt.images_path, img)
    #     image = cv2.imread(img_path)

    #     pred = crnn_recognition(image, model)
    #     res_list.append({
    #         'file_name':img,
    #         'aa':pred
    #     })
    #     i = i + 1
    #     if(i % 200==0):
    #         print(pred)
    #         print('{} samples are completed'.format(i))

    # save_dir = '/home/song/workplace/OCR/ocr_idcard/results/aa.pkl'
    # with open(save_dir, 'wb') as f:
    #     pkl.dump(res_list, f, protocol=2)
    # finished = time.time()
    # print('elapsed time: {0}'.format(finished-started))
    
