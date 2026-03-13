from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding, MultiScalePatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'
        self.prompt_type = getattr(configs, 'prompt_type', 'full')  # full | short (ablation)

        self.dropout = nn.Dropout(configs.dropout)

        if getattr(configs, 'use_multiscale_patch', False):
            self.patch_embedding = MultiScalePatchEmbedding(
                configs.d_model, configs.seq_len, self.patch_len, self.stride, configs.dropout,
                scales=getattr(configs, 'multiscale_patch_scales', None))
        else:
            self.patch_embedding = PatchEmbedding(
                configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.ablate_reprogramming = getattr(configs, 'ablate_reprogramming', False)
        if self.ablate_reprogramming:
            # 消融：去掉重编程层，用线性映射将 patch 特征直接投到 LLM 维度，不做跨模态语义对齐
            self.reprogram_proj = nn.Linear(configs.d_model, self.d_llm)
        else:
            self.word_embeddings = self.llm_model.get_input_embeddings().weight
            self.vocab_size = self.word_embeddings.shape[0]
            self.num_tokens = 1000
            self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
            self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.ablate_prompt = getattr(configs, 'ablate_prompt', False)
        self.ablate_prompt_description = getattr(configs, 'ablate_prompt_description', False)
        self.ablate_prompt_task = getattr(configs, 'ablate_prompt_task', False)
        self.ablate_prompt_stats = getattr(configs, 'ablate_prompt_stats', False)
        no_revin = getattr(configs, 'no_revin', False)
        self.normalize_layers = Normalize(configs.enc_in, affine=False, non_norm=no_revin)

        # 通道混合层：多元协同时在 reshape(B*N,T,1) 前对 (B,T,N) 做线性混合，使各通道获得其他通道信息
        self.channel_mixing = getattr(configs, 'channel_mixing', False)
        if self.channel_mixing and configs.enc_in > 1:
            self.channel_mixer = nn.Linear(configs.enc_in, configs.enc_in)
        else:
            self.channel_mixer = None

        # 辅助任务头：点对点 [B, Seq_Len, N]，不池化时间维，与数值头对齐后做逐点门控
        self.use_aux_loss = getattr(configs, 'use_aux_loss', False)
        if self.use_aux_loss:
            # 输入 block 展平为 (B*N, head_nf)，输出 (B*N, pred_len, 2) -> 对应 (B, pred_len, N) 的 P(有发运)
            self.aux_has_shipment = nn.Linear(self.head_nf, self.pred_len * 2)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_reprogramming_attention=False, return_aux_repr=False):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, return_reprogramming_attention=return_reprogramming_attention, return_aux_repr=return_aux_repr)
            if isinstance(dec_out, tuple):
                dec_out, extra = dec_out
                return dec_out[:, -self.pred_len:, :], extra
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_reprogramming_attention=False, return_aux_repr=False):
        reprogramming_attn = None
        aux_logits = None

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        if self.channel_mixer is not None:
            # (B, T, N) -> 通道维线性混合，使各通道含其他通道信息，再按通道拆成 B*N 条序列
            x_enc = self.channel_mixer(x_enc)
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        if self.ablate_prompt:
            # 消融：不要 prompt，仅用重编程后的 patch 作为 LLM 输入
            prompt_embeddings = None
        else:
            # 按需包含：数据集背景、任务指令、输入统计（三部分可单独消融）
            need_stats = not self.ablate_prompt_stats and self.prompt_type != 'short'
            if need_stats:
                min_values = torch.min(x_enc, dim=1)[0]
                max_values = torch.max(x_enc, dim=1)[0]
                medians = torch.median(x_enc, dim=1).values
                lags = self.calcute_lags(x_enc)
                trends = x_enc.diff(dim=1).sum(dim=1)

            prompt = []
            for b in range(x_enc.shape[0]):
                parts = []
                if not self.ablate_prompt_description:
                    parts.append(f"Dataset description: {self.description}")
                if not self.ablate_prompt_task:
                    parts.append(
                        f"Task description: forecast the next {self.pred_len} steps given the previous {self.seq_len} steps information; "
                    )
                if need_stats:
                    min_values_str = str(min_values[b].tolist()[0])
                    max_values_str = str(max_values[b].tolist()[0])
                    median_values_str = str(medians[b].tolist()[0])
                    lags_values_str = str(lags[b].tolist())
                    parts.append(
                        "Input statistics: "
                        f"min value {min_values_str}, "
                        f"max value {max_values_str}, "
                        f"median value {median_values_str}, "
                        f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                        f"top 5 lags are : {lags_values_str}"
                    )
                prompt.append("<|start_prompt|>" + " ".join(parts) + "<|<end_prompt>|>")

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        if not self.ablate_prompt:
            prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # CPU 上 bfloat16 与 float32 权重混用会报错，仅 CUDA 时用 bfloat16
        if x_enc.is_cuda:
            x_enc = x_enc.to(torch.bfloat16)
        enc_out, n_vars = self.patch_embedding(x_enc)
        if self.ablate_reprogramming:
            enc_out = self.reprogram_proj(enc_out)
        else:
            source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
            if return_reprogramming_attention:
                enc_out, reprogramming_attn = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings, return_attention=True)
            else:
                enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        if self.ablate_prompt:
            llama_enc_out = enc_out
        else:
            llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        block = dec_out[:, :, :, -self.patch_nums:]
        if return_aux_repr and self.use_aux_loss and hasattr(self, 'aux_has_shipment'):
            # 保留时间信息：展平 (B*N, n_vars, d_ff, patch_nums) -> (B*N, head_nf)，再映射到 (B*N, pred_len, 2)
            repr_flat = block.mean(dim=1).reshape(block.size(0), -1)
            aux_logits_flat = self.aux_has_shipment(repr_flat)
            aux_logits = {'has_shipment': aux_logits_flat.view(block.size(0), self.pred_len, 2)}
        dec_out = self.output_projection(block)
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        extra = {}
        if return_reprogramming_attention and reprogramming_attn is not None:
            extra['reprogramming_attn'] = reprogramming_attn
        if return_aux_repr and aux_logits is not None:
            extra['aux_logits'] = aux_logits
        if extra:
            return dec_out, extra
        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding, return_attention=False):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out, A = self.reprogramming(target_embedding, source_embedding, value_embedding, return_attention=return_attention)

        out = out.reshape(B, L, -1)
        out = self.out_projection(out)
        if return_attention:
            return out, A
        return out

    def reprogramming(self, target_embedding, source_embedding, value_embedding, return_attention=False):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        return reprogramming_embedding, (A if return_attention else None)
