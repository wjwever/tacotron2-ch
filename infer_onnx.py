#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

import sys
sys.path.append('waveglow/')

import numpy as np
import onnxruntime as ort
import torch
from text import text_to_sequence
from hparams import create_hparams
from torch.nn import functional as F

class OnnxEncoder:
    def __init__(self, filename):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )


    def __call__(self, x):
        logits = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
            },
        )[0]

        return logits

class ONNXDecoder:
    def __init__(self, model_path, hparams, device="cpu"):
        """
        初始化 ONNX 解码器包装器
        
        Args:
            model_path: ONNX 模型文件路径
            hparams: 模型超参数
            device: 运行设备 ('cpu' 或 'cuda')
        """
        self.hparams = hparams
        self.device = device
        
        # 创建 ONNX Runtime 会话
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers = ["CUDAExecutionProvider"] + providers
            
        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )
        
        # 获取输入输出名称
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        print(self.input_names)
        print(self.output_names)
        
        # 验证输入输出结构
        if len(self.input_names) < 2 or len(self.output_names) < 2:
            raise ValueError("模型输入输出结构不符合要求")
    
    def get_mask(self, memory):
        mask = np.ones((1, 500), dtype=bool)
        T = memory.shape[1]
        mask[:,:T] = False 
        return mask

    def get_go_frame(self, memory):
        B = memory.shape[0] 
        mel_dim = self.hparams.n_mel_channels
        return np.zeros((B, mel_dim), dtype=np.float32)

    def get_initial_state(self, batch_size=1):
        """创建初始状态"""
        hparams = self.hparams
        device = self.device
        
        return (
            np.zeros((batch_size, hparams.attention_rnn_dim), dtype=np.float32),
            np.zeros((batch_size, hparams.attention_rnn_dim), dtype=np.float32),
            np.zeros((batch_size, hparams.decoder_rnn_dim), dtype=np.float32),
            np.zeros((batch_size, hparams.decoder_rnn_dim), dtype=np.float32),
            np.zeros((batch_size, 500), dtype=np.float32),  
            np.zeros((batch_size, 500), dtype=np.float32),
            np.zeros((batch_size, hparams.encoder_embedding_dim), dtype=np.float32),
        )
    
    def __call__(self, memory, decoder_input, mask, *current_states):
        """
        执行单步推理
        
        Args:
            current_input: 当前输入 (numpy array)
            current_states: 当前状态列表 (numpy arrays)
            
        Returns:
            output: 模型输出
            new_states: 更新后的状态
        """
        # 准备输入字典
        inputs = {
            self.input_names[0]: memory, 
            self.input_names[1]: decoder_input, 
            self.input_names[2]: mask, 
        }
        
        # 添加状态输入
        for i, state_name in enumerate(self.input_names[3:]):
            inputs[state_name] = current_states[i]

        # 运行推理
        outputs = self.session.run(self.output_names, inputs)
        return outputs
        

if __name__ == "__main__":
    text = "nu2 er3 bie2 er3 de2 ye1 wa2 qing3 zhou1 yong3 kang1 zhuan3 da2 dui4 wu2 bang1 guo2 wei3 yuan2 zhang3 de5 qin1 qie4 wen4 hou4."
    text = "huan1 sheng1 xiao4 yu3 sa2 man3 cun1 zhuang1."
    text = "quan3 bi4 xu1 shuan1 yang3 huo4 juan4 yang3."
    sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
    if len(sequence) > 500:
        print("sequence length is larger than 500, please split it first")
        exit(0)

    dummy_input = torch.tensor(sequence, dtype=torch.long)
    encoder = OnnxEncoder("./encoder.onnx");
    memory = encoder(dummy_input)

    # 计算需要在第二个维度（axis=1）上填充的数量
    pad_width = ((0, 0), (0, 500 - memory.shape[1]), (0, 0))

    # 使用 numpy.pad 进行填充，填充值为 0
    padded_mem = np.pad(memory, pad_width=pad_width, mode='constant', constant_values=0.0)

    hparams = create_hparams()
    hparams.sampling_rate = 22050
    decoder = ONNXDecoder('./decoder.onnx', hparams);
    mask, decoder_input, states = decoder.get_mask(memory),  decoder.get_go_frame(memory), decoder.get_initial_state()
    
    gate_pred  = -1

    mel_out = []
    while gate_pred < 0 and len(mel_out) < 1000:
        decoder_output, gate, *states = decoder(padded_mem, decoder_input, mask, *states)
        mel_out.append(decoder_output)
        gate_pred = gate[0][0]	
        print(gate_pred, decoder_output.shape)
        decoder_input = decoder_output



    # waveglow
    waveglow_path = 'pretrain/waveglow_256channels_universal_v5.pt'
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()
    

    with torch.no_grad():
        import soundfile as sf
        mel_outputs = torch.from_numpy(np.stack(mel_out, axis=1)).cuda().transpose(1, 2)        # 直接得到 (1, n, 80)

        audio = waveglow.infer(mel_outputs, sigma=0.666)
        #ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
        audio_np = audio[0].data.cpu().numpy()  # 转为 NumPy 数组
        sf.write("audio.wav", audio_np, samplerate=hparams.sampling_rate)
