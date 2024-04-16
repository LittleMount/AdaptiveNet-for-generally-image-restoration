# 利用data数据训练网络模型
# 流程：初始化、读取数据、创建网络、训练网络

import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
import torch.nn as nn
from tqdm import tqdm
from model import AdaptiveNet
from utils import *
from torch import optim

# 初始化


# 读取数据
def imread(ot, i):
    x = cv2.imread('../dataset/data_%s/%04d.png' % (ot, i), -1)/65535
    return x[None, :, :]

def load_data(args, start=1, end=1201, shuffle=False):
    ot_set = ['0.6', '1.2', '1.8', '2.4', '2.8', '3.2', '3.6']
    input_train = np.array([imread(ot, i) for ot in ot_set for i in tqdm(range(start, end))]).astype(np.float32)
    label_train = np.array([imread('0.6', i) for i in tqdm(range(start, end))]*len(ot_set)).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(input_train), torch.from_numpy(label_train))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)
    return dl


def train(args):
    setup_logging(args.run_name)
    # 创建网络
    net = AdaptiveNet()
    net.to(args.device)
    # print(net)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    # 导入数据
    train_dl = load_data(args, shuffle=args.shuffle)    # 训练集乱序训练
    valid_dl = load_data(args, start=1201, end=1301)

    # 训练网络
    f = open('training_%s.txt' % args.run_name, 'a')
    min_loss, best_epcoh, t_loss_set, v_loss_set = 999, 0, [], []    # 用于保存最优模型和记录loss
    for epoch in range(args.epochs):
        # 训练集训练网络
        net.train()
        pbar, loss_sum = tqdm(train_dl), 0
        for i, (x, y) in enumerate(pbar):
            y_ = net(x.to(args.device))
            loss = mse(y_, y.to(args.device))
            loss_sum += loss.item()
            # 更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(MSE=loss.item())
        # 记录train_loss
        tmp_loss = loss_sum / len(train_dl)
        t_loss_set.append(tmp_loss)
        # 保存最优模型 计算验证集loss，比较大小
        net.eval()
        loss_sum = 0
        for i, (x, y) in enumerate(tqdm(valid_dl)):
            y_ = net(x.to(args.device)).clamp(0, 1)
            loss = mse(y_, y.to(args.device))
            loss_sum += loss.item()
            if i == 0:    # 保存验证集第一个结果
                save_images((y_*255).type(torch.uint8), os.path.join('results', args.run_name, f'{epoch}.png'))
        tmp_loss = loss_sum/len(valid_dl)
        if tmp_loss < min_loss:
            best_epoch = epoch
            min_loss = tmp_loss
            torch.save(net.state_dict(), os.path.join('weight', args.run_name, f'best_{args.run_name}.pt'))

        # 记录valid_loss
        v_loss_set.append(tmp_loss)

        # 打印此epoch的信息
        print(f'epoch{epoch}, t_loss: {t_loss_set[-1]:.6f}, v_loss: {v_loss_set[-1]:.6f}, best_epoch {best_epoch}: {min_loss:.6f}')
        f.write(f'epoch{epoch}, t_loss: {t_loss_set[-1]:.6f}, v_loss: {v_loss_set[-1]:.6f}, best_epoch {best_epoch}: {min_loss:.6f}\n')
    plt.plot(t_loss_set)
    plt.plot(v_loss_set)
    plt.title(f'loss_{args.run_name}.png')
    plt.savefig(f'loss_{args.run_name}.png')
    plt.show()

def test(args):
    os.makedirs(os.path.join("test results", args.run_name), exist_ok=True)
    # 创建网络
    net = AdaptiveNet()
    net.to(args.device)
    net.load_state_dict(torch.load(f'weight/{args.run_name}/best_{args.run_name}.pt'))
    net.eval()
    for i in tqdm(range(1201, 1367)):
        x = cv2.imread(f'../0. concentration genelize/dataset/data_{args.ot}/{i:04d}.png', -1)/65535
        x = torch.from_numpy(x[None, None, :, :].astype(np.float32)).to(args.device)
        y = net(x).clamp(0, 1)
        out = (y.detach().cpu().numpy().squeeze()*65535).astype(np.uint16)
        cv2.imwrite(os.path.join("test results", args.run_name, f'{i:04d}.png'), out)

def test_mix(args):
    # 创建网络
    net = AdaptiveNet()
    net.to(args.device)
    net.load_state_dict(torch.load(f'weight/{args.run_name}/best_{args.run_name}.pt'))
    net.eval()
    co_raw = np.zeros([7, 166])
    ps_raw = np.zeros([7, 166])
    co = np.zeros([7, 166])
    ps = np.zeros([7, 166])
    for j, ot in enumerate(['0.6', '1.2', '1.8', '2.4', '2.8', '3.2', '3.6']):
        os.makedirs(os.path.join("test results", args.run_name, ot), exist_ok=True)
        for i, k in tqdm(enumerate(range(1201, 1367))):
            x = cv2.imread(f'../dataset/data_{ot}/{k:04d}.png', -1)/65535
            label = cv2.imread(f'../dataset/data_0.6/{k:04d}.png', -1)/65535
            co_raw[j, i] = np.mean((x-np.mean(x))*(label-np.mean(label)))/np.std(x)/np.std(label)
            ps_raw[j, i] = 10*np.log10(1/np.mean((x-label)**2))
            x = torch.from_numpy(x[None, None, :, :].astype(np.float32)).to(args.device)
            y = net(x).clamp(0, 1)
            out = y.detach().cpu().numpy().squeeze()
            co[j, i] = np.mean((out-np.mean(out))*(label-np.mean(label)))/np.std(out)/np.std(label)
            ps[j, i] = 10*np.log10(1/np.mean((out-label)**2))
            result = (out*65535).astype(np.uint16)
            cv2.imwrite(os.path.join("test results", args.run_name, ot, f'{k:04d}_{co[j, i]:.4f}_{ps[j, i]:.4f}.png'), result)
    np.savez('corr_psnr.npz', co_raw, ps_raw, co, ps)
    print([round(np.mean(co[i, :100]), 4) for i in range(len(co))])
    print([round(np.mean(ps[i, :100]), 4) for i in range(len(ps))])
    print([round(np.std(co[i, :100]), 4) for i in range(len(co))])
    print([round(np.std(ps[i, :100]), 4) for i in range(len(ps))])
    x = np.mean(co_raw[:, 0:100], axis=1)
    y = np.mean(co[:, 0:100], axis=1)
    plt.plot(x, label='raw data'),
    plt.plot(y, label='Descatter-Net'),
    plt.legend()
    plt.show()
    plt.figure()
    x = np.mean(ps_raw[:, 0:100], axis=1)
    y = np.mean(ps[:, 0:100], axis=1)
    plt.plot(x, label='raw data'),
    plt.plot(y, label='Descatter-Net'),
    plt.legend()
    plt.show()

def test_generatation(args):
    os.makedirs(os.path.join("test results", args.run_name, 'generalization'), exist_ok=True)
    # 创建网络
    net = AdaptiveNet()
    net.to(args.device)
    net.load_state_dict(torch.load(f'weight/{args.run_name}/best_{args.run_name}.pt'))
    net.eval()
    for k in [1350, 1352, 1360]:
        x1 = cv2.imread(f'../0. concentration genelize/dataset/data_{1.2}/{k:04d}.png', -1)
        x2 = cv2.imread(f'../0. concentration genelize/dataset/data_{1.8}/{k:04d}.png', -1)
        x3 = cv2.imread(f'../0. concentration genelize/dataset/data_{2.4}/{k:04d}.png', -1)
        x4 = cv2.imread(f'../0. concentration genelize/dataset/data_{2.8}/{k:04d}.png', -1)
        x1[0:224, 224:] = x3[0:224, 224:]
        x1[224:, 0:224] = x3[224:, 0:224]
        x1[224:, 224:] = x2[224:, 224:]
        x = torch.from_numpy((x1[None, None, :, :]/65535).astype(np.float32)).to(args.device)
        y = net(x).clamp(0, 1)
        out = y.detach().cpu().numpy().squeeze()
        result = (out * 65535).astype(np.uint16)
        cv2.imwrite(os.path.join("test results", args.run_name, 'generalization', f'{k:04d}_input.png'), x1)
        cv2.imwrite(os.path.join("test results", args.run_name, 'generalization', f'{k:04d}_result.png'), result)

def classify(args):
    net = AdaptiveNet()
    net.to(args.device)
    net.load_state_dict(torch.load(f'weight/{args.run_name}/best_{args.run_name}.pt'))
    net.eval()
    feature = np.zeros([700, 256])
    for j, ot in enumerate(['0.6', '1.2', '1.8', '2.4', '2.8', '3.2', '3.6']):
        for i, k in tqdm(enumerate(range(1201, 1301))):
            x = cv2.imread(f'../0. concentration genelize/dataset/data_{ot}/{k:04d}.png', -1)/65535
            x = torch.from_numpy(x[None, None, :, :].astype(np.float32)).to(args.device)
            _, t = net(x)
            feature[j*100+i, :] = t.detach().cpu().numpy().squeeze()
    return feature


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(args.device)
    args.epochs = 200
    args.batch_size = 8
    args.image_size = 448
    args.lr = 3e-4
    args.shuffle = True
    for i in ['mix0636']:
        args.ot = i
        args.run_name = 'run_%s' % args.ot
        train(args)
        # test(args)
        # test_generatation(args)
        test_mix(args)    # 用混合训练的模型测试每一个浓度，并计算评价指标
        # feature = classify(args)
    # return feature

if __name__=='__main__':
    # feature = launch()
    launch()

'''
# 绘制散点图
for i, ot in enumerate(['0.6ml', '1.2ml', '1.8ml', '2.4ml', '2.8ml', '3.2ml', '3.6ml']):
    plt.scatter(X_pca[i*100:(i+1)*100, 0], X_pca[i*100:(i+1)*100, 1], marker='*', label=ot)
plt.legend()
plt.show()
'''