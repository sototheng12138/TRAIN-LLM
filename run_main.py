import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, vali_with_rmse, load_content
from utils.losses import ZeroInflatedLoss, MASE_loss_multivariate, JointMaskedMSEAuxBCE
from utils.auxiliary_labels import compute_derived_auxiliary_labels

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--use_multiscale_patch', action='store_true', help='use multi-scale patch embedding (ablation)')
parser.add_argument('--no_revin', action='store_true', help='ablation: disable RevIN (instance normalization)')
parser.add_argument('--ablate_reprogramming', action='store_true', help='ablation: remove reprogramming layer, use linear projection only (no text prototypes cross-attention)')
parser.add_argument('--ablate_prompt', action='store_true', help='ablation: no prompt, only reprogrammed patches as LLM input')
parser.add_argument('--ablate_prompt_description', action='store_true', help='ablation: remove dataset description from prompt')
parser.add_argument('--ablate_prompt_task', action='store_true', help='ablation: remove task instruction (e.g. forecast steps) from prompt')
parser.add_argument('--ablate_prompt_stats', action='store_true', help='ablation: remove input statistics (min/max/median/trend/lags) from prompt')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--prompt_type', type=str, default='full', choices=['full', 'short'], help='full=description+task+input stats; short=description+task only (ablation)')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Adam weight decay (L2 penalty), e.g. 1e-5')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function: MSE, ZeroInflated, MASE, JointMaskedAux (需同时 use_aux_loss)')
parser.add_argument('--zero_weight', type=float, default=2.0, help='ZeroInflated loss: weight for target==0 (default 2.0)')
parser.add_argument('--mase_freq', type=int, default=1, help='seasonal period for MASE loss (e.g. 1 for daily naive)')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--use_aux_loss', action='store_true', help='add auxiliary loss: 2-class 是否有发运 (has_shipment) for implicit regularization')
parser.add_argument('--aux_loss_weight', type=float, default=0.2, help='weight for auxiliary loss when use_aux_loss (default 0.2)')
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--multivariate', action='store_true', help='custom data: return (seq_len, enc_in) per sample for joint-window training')
parser.add_argument('--channel_mixing', action='store_true', help='use channel mixing layer in TimeLLM for cross-channel synergy (recommended with --multivariate)')

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    # TimeLLM 在 __init__ 中需要 configs.content，须在创建模型前加载
    if args.model == 'TimeLLM':
        args.content = load_content(args)
    else:
        args.content = ''

    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()

    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    if getattr(args, 'ablate_reprogramming', False):
        path = path.rstrip('/') + '_ablate_reprogram'
    if getattr(args, 'ablate_prompt', False):
        path = path.rstrip('/') + '_ablate_prompt'
    elif getattr(args, 'ablate_prompt_description', False):
        path = path.rstrip('/') + '_ablate_prompt_desc'
    elif getattr(args, 'ablate_prompt_task', False):
        path = path.rstrip('/') + '_ablate_prompt_task'
    elif getattr(args, 'ablate_prompt_stats', False):
        path = path.rstrip('/') + '_ablate_prompt_stats'
    if getattr(args, 'llm_layers', 32) == 8:
        path = path.rstrip('/') + '_ablate_llm8'
    if getattr(args, 'dropout', 0.1) == 0.15:
        path = path.rstrip('/') + '_dropout015'
    elif getattr(args, 'dropout', 0.1) == 0.05:
        path = path.rstrip('/') + '_dropout005'
    if getattr(args, 'weight_decay', 0.0) > 0:
        path = path.rstrip('/') + '_ablate_weight_decay'
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    use_aux = getattr(args, 'use_aux_loss', False) and args.model == 'TimeLLM'
    aux_w = float(getattr(args, 'aux_loss_weight', 0.2))
    if args.loss == 'JointMaskedAux':
        if not use_aux:
            raise ValueError('--loss JointMaskedAux 需同时开启 --use_aux_loss')
        criterion = JointMaskedMSEAuxBCE(lambda_weight=aux_w)
        criterion_vali = nn.MSELoss()
        accelerator.print('[JointMaskedMSEAuxBCE] 数值头纯 Masked MSE + 辅助头 BCE (两路分道扬镳，推理时用 C_aux 裁决置零), lambda = {}'.format(aux_w))
    elif args.loss == 'ZeroInflated':
        criterion = ZeroInflatedLoss(zero_weight=args.zero_weight)
        criterion_vali = criterion
        accelerator.print('[ZeroInflatedLoss] zero_weight = {} (对目标为0的样本的损失权重)'.format(args.zero_weight))
    elif args.loss == 'MASE':
        criterion = MASE_loss_multivariate(freq=args.mase_freq)
        criterion_vali = criterion
    else:
        criterion = nn.MSELoss()
        criterion_vali = criterion
    mae_metric = nn.L1Loss()
    if use_aux and args.loss != 'JointMaskedAux':
        criterion_ce = nn.CrossEntropyLoss()
        accelerator.print('[AuxLoss] enabled, aux_loss_weight = {}'.format(aux_w))

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, batch in tqdm(enumerate(train_loader)):
            if len(batch) == 5:
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_feat_ids = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_feat_ids = None
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)

            # encoder - decoder
            f_dim = -1 if args.features == 'MS' else 0
            batch_y_future = batch_y[:, -args.pred_len:, f_dim:]
            # 归一化陷阱：batch_y_future 来自 DataLoader，若 scale=True 则为 StandardScaler 后数据；0 吨会变负值
            # 辅助标签与 Masked MSE 的 mask 必须用「原始量纲」判有/无货。独立通道时 batch_y_future 为 (B,pred_len,1)，scaler 按 4 列 fit，需按通道逆变换
            raw_batch_y_future = None
            _ds = getattr(train_loader, 'dataset', None)
            if _ds is not None and getattr(_ds, 'scale', False) and hasattr(_ds, 'scaler'):
                b, l, n = batch_y_future.shape
                scale_ = _ds.scaler.scale_
                mean_ = _ds.scaler.mean_
                if n == 1 and batch_feat_ids is not None:
                    # 独立通道：(B, pred_len, 1)，每样本对应一通道，用该通道的 mean_/scale_ 逆变换
                    raw_batch_y_future = torch.empty(b, l, 1, dtype=torch.float32, device=batch_y_future.device)
                    for bi in range(b):
                        c = int(batch_feat_ids[bi].item() if hasattr(batch_feat_ids[bi], 'item') else batch_feat_ids[bi])
                        raw_batch_y_future[bi, :, 0] = torch.from_numpy(
                            (batch_y_future[bi, :, 0].cpu().numpy() * scale_[c]) + mean_[c]
                        ).float().to(batch_y_future.device)
                elif n == scale_.shape[0]:
                    flat = batch_y_future.detach().cpu().numpy().reshape(-1, n)
                    raw_flat = _ds.inverse_transform(flat)
                    raw_batch_y_future = torch.from_numpy(raw_flat.reshape(b, l, n).astype(np.float32)).to(batch_y_future.device)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if use_aux and (not args.output_attention):
                        out_pack = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_aux_repr=True)
                        if isinstance(out_pack, tuple):
                            outputs, extra = out_pack
                            aux_logits = extra.get('aux_logits')
                        else:
                            outputs = out_pack
                            aux_logits = None
                    else:
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        aux_logits = None

                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y_future.to(accelerator.device)
                    if args.loss == 'JointMaskedAux' and use_aux and aux_logits is not None:
                        _y_aux = raw_batch_y_future if raw_batch_y_future is not None else batch_y_future
                        aux_labels = compute_derived_auxiliary_labels(_y_aux, enc_in=args.enc_in, point_to_point=True)
                        loss = criterion(outputs, aux_logits['has_shipment'], batch_y, aux_labels['has_shipment'], raw_targets_num=raw_batch_y_future)
                    else:
                        if args.loss == 'MASE':
                            loss = criterion(batch_x, outputs, batch_y, torch.ones_like(batch_y))
                        else:
                            loss = criterion(outputs, batch_y)
                        if use_aux and (aux_logits is not None):
                            _y_aux = raw_batch_y_future if raw_batch_y_future is not None else batch_y_future
                            aux_labels = compute_derived_auxiliary_labels(_y_aux, enc_in=args.enc_in)
                            aux_log = aux_logits['has_shipment']
                            if aux_log.dim() == 3:
                                aux_log = aux_log.mean(dim=1)
                            lab = aux_labels['has_shipment']
                            if lab.dim() == 1 and aux_log.size(0) == lab.size(0) * args.enc_in:
                                lab = lab.repeat_interleave(args.enc_in, dim=0)
                            aux_loss = criterion_ce(aux_log, lab)
                            loss = loss + aux_w * aux_loss
                    train_loss.append(loss.item())
            else:
                f_dim = -1 if args.features == 'MS' else 0
                batch_y_future = batch_y[:, -args.pred_len:, f_dim:]

                if use_aux and (not args.output_attention):
                    out_pack = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_aux_repr=True)
                    if isinstance(out_pack, tuple):
                        outputs, extra = out_pack
                        aux_logits = extra.get('aux_logits')
                    else:
                        outputs = out_pack
                        aux_logits = None
                else:
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    aux_logits = None

                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y_future
                if args.loss == 'JointMaskedAux' and use_aux and aux_logits is not None:
                    _y_aux = raw_batch_y_future if raw_batch_y_future is not None else batch_y_future
                    aux_labels = compute_derived_auxiliary_labels(_y_aux, enc_in=args.enc_in, point_to_point=True)
                    loss = criterion(outputs, aux_logits['has_shipment'], batch_y, aux_labels['has_shipment'], raw_targets_num=raw_batch_y_future)
                else:
                    if args.loss == 'MASE':
                        loss = criterion(batch_x, outputs, batch_y, torch.ones_like(batch_y))
                    else:
                        loss = criterion(outputs, batch_y)
                    if use_aux and (aux_logits is not None):
                        _y_aux = raw_batch_y_future if raw_batch_y_future is not None else batch_y_future
                        aux_labels = compute_derived_auxiliary_labels(_y_aux, enc_in=args.enc_in)
                        aux_log = aux_logits['has_shipment']
                        if aux_log.dim() == 3:
                            aux_log = aux_log.mean(dim=1)
                        lab = aux_labels['has_shipment']
                        if lab.dim() == 1 and aux_log.size(0) == lab.size(0) * args.enc_in:
                            lab = lab.repeat_interleave(args.enc_in, dim=0)
                        aux_loss = criterion_ce(aux_log, lab)
                        loss = loss + aux_w * aux_loss
                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss)
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion_vali, mae_metric)
        test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion_vali, mae_metric)
        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

accelerator.wait_for_everyone()
# 用保存的 best checkpoint 在测试集上算一次，直接得到最终 Test Loss / MAE / RMSE
ckpt_file = os.path.join(path, 'checkpoint')
if os.path.exists(ckpt_file):
    state = torch.load(ckpt_file, map_location=accelerator.device)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.load_state_dict(state, strict=True)
    test_loss_final, test_mae_final, test_rmse_final = vali_with_rmse(
        args, accelerator, model, test_loader, criterion_vali, mae_metric)
    if accelerator.is_local_main_process:
        accelerator.print('=' * 60)
        accelerator.print('Best checkpoint (by Vali Loss) on Test Set:')
        accelerator.print('  Test Loss: {:.7f}  |  MAE: {:.7f}  |  RMSE: {:.7f}'.format(
            test_loss_final, test_mae_final, test_rmse_final))
        accelerator.print('=' * 60)
# 不再在训练结束后删除 checkpoints，保留 EarlyStopping 保存的最佳模型