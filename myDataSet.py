#coding=utf-8

from torch.utils.data import Dataset
import torch
import load_data as ld
import numpy as np


k = 3
stride = 1
print("kmer", k)
print("stride", stride)

#只包含原始输入
class MyDataSet(Dataset):
    def __init__(self, input, label, model):
        self.input_seq = input
        self.output = label
        self.model = model

    def __getitem__(self, index):
        input_seq_origin = self.input_seq[index]
        # #原先是加T，换成不加T
        # #加T给conv
        input_seq = np.array(ld.k_mer_stride(input_seq_origin, k, stride)).T    #(100, 99) 
        input_seq_OH = np.array(ld.OH_mer_stride(input_seq_origin, k, stride))  #(99, 64)

        input_seq = torch.from_numpy(input_seq).type(torch.FloatTensor).cuda()
        # # # vec 和 OH拼接
        # input_seq = torch.from_numpy(input_seq).type(torch.FloatTensor)
        input_seq_OH = torch.from_numpy(input_seq_OH).type(torch.FloatTensor).cuda()

        input_seq = input_seq.unsqueeze(0)  #(1, 100, 99)
        input_seq = self.model.embedding(input_seq).transpose(1,2)  #(1, 99, 64)
        input_seq = input_seq.squeeze(0)  #(99, 64)
        # #直接相加
        # input_seq = input_seq + input_seq_OH
        # # # # #横向扩展
        input_seq = torch.cat((input_seq, input_seq_OH), 1) #(128,99)
        input_seq = input_seq.transpose(0,1)
        # # ##竖向扩展
        # # input_seq = torch.cat((input_seq, input_seq_OH), 0) #(198, 64)
        # # # print(input_seq.shape)

        output_seq = self.output[index]
        output_seq = torch.Tensor([output_seq]).cuda()
        output_seq = torch.Tensor([output_seq])
        return input_seq, output_seq

    def __len__(self):
        return len(self.input_seq)

