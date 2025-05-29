#!/usr/bin/env python
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import time
import glob
import pickle
import random
import traceback
import resource

from collections import OrderedDict

import apex
import torch
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from args import get_parser
from loss import LabelSmoothingCrossEntropy, get_mmd_loss, KLLoss, KL_divergence
from model.infogcn import InfoGCN, TextCLIP, Encoder, Decoder
import torch.nn as nn
from utils import get_vector_property
from utils import BalancedSampler as BS
from utils import gen_label, create_logits, reparameterize
from Text_Prompt import *
from torch.cuda.amp import GradScaler, autocast

# 注意学习率问题 ntu120为0.1，ntu60为0.01
# python main.py --half=True --batch_size=96 --test_batch_size=96 \
#     --step 50 60 --num_epoch=70 --n_heads=3 --num_worker=4 --k=8 \
#     --dataset=ntu --num_class=60 --lambda_1=1e-4 --lambda_2=1e-1 --z_prior_gain=3 \
#     --use_vel=False --datacase=NTU60_CS --weight_decay=0.0005 \
#     --num_person=2 --num_point=25 --graph=graph.ntu_rgb_d.Graph --feeder=feeders.feeder_ntu.Feeder
#时刻确认模型框架的完整性
prediction_ratio = 0.2
seg_method = 'ntu_seg_1'
# GAP
classes, num_text_aug, text_dict = text_prompt_openai_pasta_pool_4part()
text_list = text_prompt_openai_random()


# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)   # 为所有可用的GPU设置随机种子
    torch.manual_seed(seed)  # 为CPU上的PyTorch操作设置随机种子
    np.random.seed(seed)   # 为numpy内置的random模块设置随机种子
    random.seed(seed)   #  为Python内置的random模块设置随机种子
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')  # 将字符串 import_str 从右向左分割成三部分，以最后一个 . 为分隔符
    __import__(mod_str)  # 将模块加载到 sys.modules 中
    try:
        return getattr(sys.modules[mod_str], class_str)  # 从已导入的模块中获取指定的类，sys.modules 是一个字典，键是模块名，值是模块对象
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


class Processor():
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.global_step = 0
        # pdb.set_trace()
        self.load_model()
        self.scaler = GradScaler(enabled=self.arg.half)
        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.best_acc = 0
        self.best_acc_epoch = 0

        model = self.model.cuda()

        # if self.arg.half:
        #     self.model, self.optimizer = apex.amp.initialize(
        #         model,
        #         self.optimizer,
        #         opt_level=f'O{self.arg.amp_opt_level}'   # 混合精度训练fp16 fp32
        #     ) 
        #     if self.arg.amp_opt_level != 1:
        #         self.print_log('[WARN] nn.DataParallel is not yet supported by amp_opt_level != "O1"')

        # self.model = torch.nn.DataParallel(model, device_ids=(0,1,2))

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        print("begin to load data")
        self.data_loader = dict()
        data_path = f'/data/data1/zhengchaoyang/infogcn_GAP/data/{seg_method}/{self.arg.datacase}_{prediction_ratio}_aligned.npz'
        # data_path = f'data/{seg_method}/{self.arg.datacase}_{prediction_ratio}_aligned.npz'
        if self.arg.phase == 'train':
            dt = Feeder(data_path=data_path,
                split='train',
                window_size=64,
                p_interval=[0.5, 1],
                vel=self.arg.use_vel,
                random_rot=self.arg.random_rot,
                sort=True if self.arg.balanced_sampling else False,
            )
            print("finish Feeder")
            if self.arg.balanced_sampling:
                sampler = BS(data_source=dt, args=self.arg)  # 数据加载时，确保每个批次中每个类别的样本数量是平衡的
                shuffle = False
            else:
                sampler = None
                shuffle = True
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=dt,
                sampler=sampler,
                batch_size=self.arg.batch_size,
                shuffle=shuffle,
                num_workers=self.arg.num_worker,
                drop_last=True,
                pin_memory=True,
                worker_init_fn=init_seed)
            print("finish data_loader['train']")
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(
                data_path=data_path,
                split='test',
                window_size=64,
                p_interval=[0.95],
                vel=self.arg.use_vel
            ),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=init_seed)
        print("load data successfully")

    def load_model(self):
        self.model = InfoGCN(
            num_class=self.arg.num_class,
            num_point=self.arg.num_point,
            num_person=self.arg.num_person,
            graph=self.arg.graph,
            in_channels=3,
            drop_out=0,
            num_head=self.arg.n_heads,
            k=self.arg.k,
            noise_ratio=self.arg.noise_ratio,
            gain=self.arg.z_prior_gain
        )

        # VAE 
        if self.arg.dataset == 'ntu':
            semantic_latent_size = 160
            style_latent_size = 8
            vis_emb_input_size = 256
            text_emb_input_size = 512
        elif self.arg.dataset == 'ntu120':
            semantic_latent_size = 512
            style_latent_size = 32
            vis_emb_input_size = 256
            text_emb_input_size = 512
        self.sequence_encoder = Encoder(
            [vis_emb_input_size, semantic_latent_size + style_latent_size], style_latent_size).cuda()
        self.sequence_decoder = Decoder(
            [semantic_latent_size + style_latent_size, vis_emb_input_size]).cuda()
        self.text_encoder = Encoder(
            [text_emb_input_size, semantic_latent_size]).cuda()
        self.text_decoder = Decoder(
            [semantic_latent_size, text_emb_input_size]).cuda()
        

        self.loss = LabelSmoothingCrossEntropy().cuda()
        self.klloss = KLLoss().cuda()

        # VAE_Loss
        self.mse_criterion = nn.MSELoss().cuda()
        self.bce_criterion = nn.BCELoss().cuda()
        # self.KL_divergence = KL_divergence().cuda()


        self.model_text_dict = nn.ModuleDict()
        self.arg.model_args['head'] = ['ViT-B/32']
        for name in self.arg.model_args['head']:
            model_, preprocess = clip.load(name)
            # model_, preprocess = clip.load('ViT-L/14', device)
            del model_.visual
            model_text = TextCLIP(model_)
            model_text = model_text.cuda()
            self.model_text_dict[name] = model_text

        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)
            # 有序字典
            weights = OrderedDict([[k.split('module.')[-1], v.cuda()] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            params = []
            for model in [self.model, self.sequence_encoder, self.sequence_decoder, self.text_encoder, self.text_decoder]:
                params += list(model.parameters())
            self.optimizer = optim.SGD(
                params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            params = []
            for model in [self.model, self.sequence_encoder, self.sequence_decoder, self.text_encoder, self.text_decoder]:
                params += list(model.parameters())
            self.optimizer = optim.Adam(
                params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch and self.arg.weights is None:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        self.adjust_learning_rate(epoch)

        loss_value = []
        mmd_loss_value = []
        l2_z_mean_value = []
        acc_value = []
        cos_z_value = []
        dis_z_value = []
        cos_z_prior_value = []
        dis_z_prior_value = []

        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        for i, (data, y, index) in enumerate(tqdm(self.data_loader['train'], dynamic_ncols=True)):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda()
                y = y.long().cuda()
            timer['dataloader'] += self.split_time()

            # forward
            with autocast(enabled=self.arg.half):
                y_hat, z, x, logit_scale, part_feature_list = self.model(data)

                # add GAP and VAE and compute loss
                label = y.long().cuda()
                VAE_loss_list = []

                # VAE_loss cofficients
                # kld_loss_factor = max(
                #     (0.1 * (i - (len(self.data_loader['train']) / 1700 * 1000)) / (len(self.data_loader['train']) / 1700 * 3000)), 0)
                # kld_loss_factor_2 = max(
                #     (0.1 * (i - int(len(self.data_loader['train']) / 3)) / (len(self.data_loader['train']) / 1700 * 3000)), 0) * (epoch > 1)
                # cross_alignment_loss_factor = 1 * (i > int(0.8 * len(self.data_loader['train'])))

                if i <= int(len(self.data_loader['train']) / 3):
                    kld_loss_factor = 0
                    kld_loss_factor_2 = 0
                else:
                    kld_loss_factor = 1.5 * \
                        (float(i) / len(self.data_loader['train']) - 1/3) * 0.023
                    kld_loss_factor_2 = 1.5 * \
                        (float(i) / len(self.data_loader['train']) - 1/3) * 0.011
                cross_alignment_loss_factor = 1 * (i > int(0.8 * len(self.data_loader['train'])))


                # GAP 4parts text feature
                for ind in range(num_text_aug):
                    if ind > 0:
                        # 根据标签和当前的文本模板索引ind，从 text_dict 中提取对应的文本嵌入并将这些嵌入堆叠成一个批量张量。
                        text_id = np.ones(len(label),dtype=np.int8) * ind
                        texts = torch.stack([text_dict[j][i,:] for i,j in zip(label,text_id)])
                        texts = texts.cuda()

                        s = part_feature_list[ind-1].cuda()
                    else:
                        # 为每个label随即挑选一个近义词的编码
                        continue
                        # texts = list()
                        # for i in range(len(label)):
                        #     #test_list label的近义词经过clip编码
                        #     text_len = len(text_list[label[i]])
                        #     text_id = np.random.randint(text_len,size=1)
                        #     text_item = text_list[label[i]][text_id.item()]
                        #     texts.append(text_item)
                    
                        # texts = torch.cat(texts).cuda()

                        # s = x.cuda()


                    # [batch, 512]
                    text_embedding = self.model_text_dict[self.arg.model_args['head'][0]](texts).float()

                    # VAE
                    t = text_embedding.cuda()

                    smu, slv, ismu, islv = self.sequence_encoder(s, instance_style=True)
                    sz = reparameterize(smu, slv)
                    isz = reparameterize(ismu, islv)
                    sout = self.sequence_decoder(torch.cat([sz, isz], dim=-1))

                    tmu, tlv = self.text_encoder(t)
                    tz = reparameterize(tmu, tlv)
                    tout = self.text_decoder(tz)

                    sfromt = self.sequence_decoder(torch.cat([tz, isz], dim=-1))
                    tfroms = self.text_decoder(sz)

                    # ELBO Loss
                    loss_rss = self.mse_criterion(s, sout)
                    loss_rtt = self.mse_criterion(t, tout)
                    loss_kld_s = KL_divergence(smu, slv).cuda()
                    loss_kld_is = KL_divergence(ismu, islv).cuda()
                    loss_kld_t = KL_divergence(tmu, tlv).cuda()

                    # Cross Alignment Loss
                    loss_rst = self.mse_criterion(s, sfromt)
                    loss_rts = self.mse_criterion(t, tfroms)

                    VAE_loss = loss_rss + loss_rtt
                    VAE_loss -= kld_loss_factor * (loss_kld_s + loss_kld_is) + \
                        kld_loss_factor_2 * loss_kld_t
                    VAE_loss += cross_alignment_loss_factor * (loss_rst + loss_rts)
                    VAE_loss_list.append(VAE_loss)


                # infogcn loss compute
                mmd_loss, l2_z_mean, z_mean = get_mmd_loss(z, self.model.z_prior, y, self.arg.num_class)
                cos_z, dis_z = get_vector_property(z_mean)
                cos_z_prior, dis_z_prior = get_vector_property(self.model.z_prior)
                cos_z_value.append(cos_z.data.item())
                dis_z_value.append(dis_z.data.item())
                cos_z_prior_value.append(cos_z_prior.data.item())
                dis_z_prior_value.append(dis_z_prior.data.item())

                cls_loss = self.loss(y_hat, y)
                loss = self.arg.lambda_2* mmd_loss + self.arg.lambda_1* l2_z_mean + cls_loss + 0.001 * sum(VAE_loss_list)

            # graph = make_dot(loss)
            # graph.render("mygraph", format="pdf")
            # backward
            self.optimizer.zero_grad()
            # 使用 scaler 缩放梯度
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # if self.arg.half:
            #     with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            #     loss.backward()

            #self.optimizer.step()

            loss_value.append(cls_loss.data.item())
            mmd_loss_value.append(mmd_loss.data.item())
            l2_z_mean_value.append(l2_z_mean.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(y_hat.data, 1)
            acc = torch.mean((predict_label == y.data).float())
            acc_value.append(acc.data.item())

            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(f'\tTraining loss: {np.mean(loss_value):.4f}.  Training acc: {np.mean(acc_value)*100:.2f}%.')
        self.print_log(f'\tTime consumption: [Data]{proportion["dataloader"]}, [Network]{proportion["model"]}')

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            #torch.save(weights, f'{self.arg.work_dir}/runs-{epoch+1}-{int(self.global_step)}.pt')
            torch.save(weights, f'{self.arg.work_dir}/runs-{epoch+1}.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], save_z=False):
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            cls_loss_value = []
            mmd_loss_value = []
            l2_z_mean_value = []
            score_frag = []
            label_list = []
            pred_list = []
            cos_z_value = []
            dis_z_value = []
            cos_z_prior_value = []
            dis_z_prior_value = []
            step = 0
            z_list = []
            for data, y, index in tqdm(self.data_loader[ln], dynamic_ncols=True):
                label_list.append(y)
                with torch.no_grad():
                    data = data.float().cuda()
                    y = y.long().cuda()
                    # forward
                    with autocast(enabled=self.arg.half):
                        y_hat, z, feature_dict, logit_scale, part_feature_list = self.model(data)
                    if save_z:
                        z_list.append(z.data.cpu().numpy())
                    mmd_loss, l2_z_mean, z_mean = get_mmd_loss(z, self.model.z_prior, y, self.arg.num_class)
                    cos_z, dis_z = get_vector_property(z_mean)
                    cos_z_prior, dis_z_prior = get_vector_property(self.model.z_prior)
                    cos_z_value.append(cos_z.data.item())
                    dis_z_value.append(dis_z.data.item())
                    cos_z_prior_value.append(cos_z_prior.data.item())
                    dis_z_prior_value.append(dis_z_prior.data.item())
                    cls_loss = self.loss(y_hat, y)
                    loss = self.arg.lambda_2*mmd_loss + self.arg.lambda_1*l2_z_mean + cls_loss
                    score_frag.append(y_hat.data.cpu().numpy())
                    loss_value.append(loss.data.item())
                    cls_loss_value.append(cls_loss.data.item())
                    mmd_loss_value.append(mmd_loss.data.item())
                    l2_z_mean_value.append(l2_z_mean.data.item())

                    _, predict_label = torch.max(y_hat.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            cls_loss = np.mean(cls_loss_value)
            mmd_loss = np.mean(mmd_loss_value)
            l2_z_mean_loss = np.mean(l2_z_mean_value)
            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {:4f}.'.format(
                ln, self.arg.n_desired//self.arg.batch_size, np.mean(cls_loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1
                with open(f'{self.arg.work_dir}/best_score.pkl', 'wb') as f:
                    pickle.dump(score_dict, f)

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)

            if save_z:
                z_list = np.concatenate(z_list)
                np.savez(f'{self.arg.work_dir}/z_values.npz', z=z_list, z_prior=self.model.z_prior.cpu().numpy(), y=label_list)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = 0
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)+  count_parameters(self.sequence_encoder)+ count_parameters(self.sequence_decoder)+ count_parameters(self.text_encoder)+ count_parameters(self.text_decoder)}.'
                            )
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                # save_model = (epoch + 1 == self.arg.num_epoch) and (epoch + 1 > self.arg.save_epoch)
                save_model = ((epoch + 1)%10 ==0) 
                self.train(epoch, save_model=save_model)

                # if epoch > 80:
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+f'{self.arg.num_epoch}'+'.pt'))[0] # +str(self.best_acc_epoch)
            print(weights_path)
            weights = torch.load(weights_path)
            self.model.load_state_dict(weights)

            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'])
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            #self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], save_z=True)
            self.print_log('Done.\n')

def main():
    # parser arguments
    parser = get_parser()
    arg = parser.parse_args()
    arg.work_dir = f"results_0.001_seg1_oldtext/{arg.dataset}_{arg.datacase}_{arg.k}_{prediction_ratio}"
    init_seed(arg.seed)
    # execute process
    processor = Processor(arg)
    processor.start()

if __name__ == '__main__':
    main()
