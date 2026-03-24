from torch.utils.data import DataLoader

# 1. 在这里的 import 列表中，加上你刚才创建的 Dataset_Custom_Iron
# (假设你把 Dataset_Custom_Iron 写在了 data_provider_pretrain.data_loader 文件中)
from data_provider_pretrain.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom_Iron

# 2. 在数据字典中，增加 'custom' 映射
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom_Iron,   # <--- 加上这一行！
}

def data_provider(args, data, data_path, pretrain=True, flag='train'):
    # 当你在 bash 脚本中写 --data custom 时，这里的 data 就是 'custom'
    # 框架就会从字典里抓取 Dataset_Custom_Iron 来处理数据
    Data = data_dict[data]  
    
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        seasonal_patterns=args.seasonal_patterns,
        pretrain=pretrain
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader