import torch
import torch.nn as nn
from model import Tacotron2
from train import load_model
from hparams import create_hparams
import numpy as np
from text import text_to_sequence
from torch.nn import functional as F
import sys

class Encoder(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
    
    def forward(self, text):
        embedded_inputs = self.model.embedding(dummy_input).transpose(1, 2)
        encoder_outputs = self.model.encoder.inference(embedded_inputs)
        return encoder_outputs

class Decoder(nn.Module):
    def __init__(self, original_model, hparams):
        super().__init__()
        self.model = original_model
        self.hparams = hparams

    def get_go_frame(self, memory):
        B = memory.size(0) 
        mel_dim = self.hparams.n_mel_channels
        return torch.zeros(B, mel_dim)

    def get_mask(self, memory):
        mask = torch.ones((1, 500), dtype=torch.bool)
        T = memory.size(1)
        mask[:,:T] = False 
        return mask

    def get_initial_state(self, batch_size=1, device="cpu"):
        """创建初始状态元组"""
        hparams = self.hparams
        return (
            torch.zeros(batch_size, hparams.attention_rnn_dim, device=device),
            torch.zeros(batch_size, hparams.attention_rnn_dim, device=device),
            torch.zeros(batch_size, hparams.decoder_rnn_dim, device=device),
            torch.zeros(batch_size, hparams.decoder_rnn_dim, device=device),
            torch.zeros(batch_size, 500, device=device),  # MAX_TIME=500
            torch.zeros(batch_size, 500, device=device),
            torch.zeros(batch_size, hparams.encoder_embedding_dim, device=device)
        )

    def init(self):
        B = 1
        MAX_TIME = 500
        hparams = self.hparams

        self.attention_hidden = torch.zeros(B, hparams.attention_rnn_dim)
        self.attention_cell = torch.zeros(B, hparams.attention_rnn_dim)
        self.decoder_hidden = torch.zeros(B, hparams.decoder_rnn_dim)
        self.decoder_cell = torch.zeros(B, hparams.decoder_rnn_dim)
        self.attention_weights = torch.zeros(B, MAX_TIME)
        self.attention_weights_cum = torch.zeros(B, MAX_TIME)
        self.attention_context = torch.zeros(B, hparams.encoder_embedding_dim)

        #self.memory = memory
        #self.processed_memory = self.attention_layer.memory_layer(memory)
        #self.mask = torch.zeros(1, MAX_TIME)

        
    def forward(self, memory, decoder_input, mask, *states):
         # 解包状态
        (attention_hidden, attention_cell, 
         decoder_hidden, decoder_cell,
         attention_weights, attention_weights_cum,
         attention_context) = states

        decoder_input = self.model.decoder.prenet(decoder_input)
        processed_memory = self.model.decoder.attention_layer.memory_layer(memory)
        cell_input = torch.cat((decoder_input, attention_context), -1)
        attention_hidden, attention_cell = self.model.decoder.attention_rnn(
            cell_input, (attention_hidden, attention_cell))

        attention_weights_cat = torch.cat(
            (attention_weights.unsqueeze(1),
             attention_weights_cum.unsqueeze(1)), dim=1)

        attention_context, attention_weights = self.model.decoder.attention_layer(
            attention_hidden, memory, processed_memory,
            attention_weights_cat, mask)

        attention_weights_cum += attention_weights
        decoder_input = torch.cat(
            (attention_hidden, attention_context), -1)
        decoder_hidden, decoder_cell = self.model.decoder.decoder_rnn(
            decoder_input, (decoder_hidden, decoder_cell))

        decoder_hidden_attention_context = torch.cat(
            (decoder_hidden, attention_context), dim=1)

        decoder_output = self.model.decoder.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.model.decoder.gate_layer(decoder_hidden_attention_context)
         # 创建新状态元组
        new_states = (
            attention_hidden, attention_cell,
            decoder_hidden, decoder_cell,
            attention_weights, attention_weights_cum,
            attention_context
        )
        return decoder_output, gate_prediction, new_states


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python export_onnx.py checkpoint")
        exit(0)
    checkpoint_path = sys.argv[1];
    hparams = create_hparams()
    hparams.use_cuda = False
    model = load_model(hparams)
    model.eval()
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train(False)  # 显式设置为评估模式

    text = "nu2 er3 bie2 er3 de2 ye1 wa2 qing3 zhou1 yong3 kang1 zhuan3 da2 dui4 wu2 bang1 guo2 wei3 yuan2 zhang3 de5 qin1 qie4 wen4 hou4."
    sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
    dummy_input = torch.tensor(sequence, dtype=torch.long)
    encoder = Encoder(model).eval()
    memory = encoder(dummy_input)

#decoder
    decoder = Decoder(model, hparams).eval()
    mask, decoder_input, states = decoder.get_mask(memory), decoder.get_go_frame(memory), decoder.get_initial_state()


    target_length = 500
    current_length = memory.size(1)
    pad_length = target_length - current_length
# 使用 F.pad 进行填充
# (0, 0) 表示不填充最后一个维度（512）
# (0, pad_length) 表示在第二个维度（→ 500）右侧填充
    padded_mem = F.pad(memory, (0, 0, 0, pad_length), mode='constant', value=0)
# 计算需要填充的长度（在第二个维度上填充）

    gate_pred  = -1
    while gate_pred < 0:
        decoder_output, gate_pred, states = decoder(padded_mem, decoder_input, mask, *states)
        decoder_input = decoder_output
        print(gate_pred)

# 输入名称（包括所有状态）
    input_names = ["memory", "decoder_input", "mask"] + [
        "attention_hidden", "attention_cell",
        "decoder_hidden", "decoder_cell",
        "attention_weights", "attention_weights_cum",
        "attention_context"
    ]

# 输出名称
    output_names = ["decoder_out", "gate_pred"] + [
        "new_attention_hidden", "new_attention_cell",
        "new_decoder_hidden", "new_decoder_cell",
        "new_attention_weights", "new_attention_weights_cum",
        "new_attention_context"
    ]

    torch.onnx.export(
        decoder,
        (padded_mem, decoder_input, mask, *states),
        "decoder.onnx",
        input_names= input_names,
        output_names= output_names,
        dynamic_axes= {
            **{name: {0: "batch_size"} for name in input_names},
            **{name: {0: "batch_size"} for name in output_names}
        },
        opset_version=11,  # 推荐opset
        verbose=False
    )

    input_names = ["input_text"]
    output_names = ["encoder_outputs"]
    torch.onnx.export(
        encoder,
        dummy_input,
        "encoder.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
           "input_text": {0: "batch", 1: "text_length"},          # 可变文本长度
            "encoder_outputs" : {0:"batch", 1:"text_length"},
        },
        opset_version=10,  # 推荐opset
        verbose=True
    )
    print("checkout encoder.onnx decoder.onnx")
