import torch
import torch.nn as nn
import numpy as np
import math
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from emotalk.wav2vec import Wav2Vec2Model, Wav2Vec2ForSpeechClassification
from emotalk.utils import init_biased_mask, enc_dec_mask


class EmoTalk(nn.Module):
    def __init__(self, args):
        super(EmoTalk, self).__init__()
        self.feature_dim = args.feature_dim
        self.bs_dim = args.bs_dim
        self.device = args.device
        self.batch_size = args.batch_size
        self.audio_encoder_cont = Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.audio_encoder_cont.feature_extractor._freeze_parameters()
        self.audio_encoder_emo = Wav2Vec2ForSpeechClassification.from_pretrained(
            "r-f/wav2vec-english-speech-emotion-recognition")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "r-f/wav2vec-english-speech-emotion-recognition")
        self.audio_encoder_emo.wav2vec2.feature_extractor._freeze_parameters()
        self.max_seq_len = args.max_seq_len
        self.audio_feature_map_cont = nn.Linear(1024, 512)
        self.audio_feature_map_emo = nn.Linear(1024, 832)
        self.audio_feature_map_emo2 = nn.Linear(832, 256)
        self.relu = nn.ReLU()
        self.biased_mask1 = init_biased_mask(n_head=4, max_seq_len=args.max_seq_len, period=args.period)
        self.one_hot_level = np.eye(2)
        self.obj_vector_level = nn.Linear(2, 32)
        self.one_hot_person = np.eye(24)
        self.obj_vector_person = nn.Linear(24, 32)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=args.feature_dim,
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.bs_map_r = nn.Linear(self.feature_dim, self.bs_dim)
        nn.init.constant_(self.bs_map_r.weight, 0)
        nn.init.constant_(self.bs_map_r.bias, 0)

    def forward(self, data):
        frame_num11 = data["target11"].shape[1]
        frame_num12 = data["target12"].shape[1]
        inputs12 = self.processor(torch.squeeze(data["input12"]), sampling_rate=16000, return_tensors="pt",
                                  padding="longest").input_values.to(self.device)
        hidden_states_cont1 = self.audio_encoder_cont(inputs12, frame_num=frame_num11).last_hidden_state
        hidden_states_cont12 = self.audio_encoder_cont(inputs12, frame_num=frame_num12).last_hidden_state
        inputs21 = self.feature_extractor(torch.squeeze(data["input21"]), sampling_rate=16000, padding=True,
                                          return_tensors="pt").input_values.to(self.device)
        inputs12 = self.feature_extractor(torch.squeeze(data["input12"]), sampling_rate=16000, padding=True,
                                          return_tensors="pt").input_values.to(self.device)

        output_emo1 = self.audio_encoder_emo(inputs21, frame_num=frame_num11)
        output_emo2 = self.audio_encoder_emo(inputs12, frame_num=frame_num12)

        hidden_states_emo1 = output_emo1.hidden_states
        hidden_states_emo2 = output_emo2.hidden_states

        label1 = output_emo1.logits
        onehot_level = self.one_hot_level[data["level"]]
        onehot_level = torch.from_numpy(onehot_level).to(self.device).float()
        onehot_person = self.one_hot_person[data["person"]]
        onehot_person = torch.from_numpy(onehot_person).to(self.device).float()
        if data["target11"].shape[0] == 1:
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0)
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0)
        else:
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0).permute(1, 0, 2)
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0).permute(1, 0, 2)

        obj_embedding_level11 = obj_embedding_level.repeat(1, frame_num11, 1)
        obj_embedding_level12 = obj_embedding_level.repeat(1, frame_num12, 1)
        obj_embedding_person11 = obj_embedding_person.repeat(1, frame_num11, 1)
        obj_embedding_person12 = obj_embedding_person.repeat(1, frame_num12, 1)
        hidden_states_cont1 = self.audio_feature_map_cont(hidden_states_cont1)
        hidden_states_emo11_832 = self.audio_feature_map_emo(hidden_states_emo1)
        hidden_states_emo11_256 = self.relu(self.audio_feature_map_emo2(hidden_states_emo11_832))

        hidden_states11 = torch.cat(
            [hidden_states_cont1, hidden_states_emo11_256, obj_embedding_level11, obj_embedding_person11], dim=2)
        hidden_states_cont12 = self.audio_feature_map_cont(hidden_states_cont12)
        hidden_states_emo12_832 = self.audio_feature_map_emo(hidden_states_emo2)
        hidden_states_emo12_256 = self.relu(self.audio_feature_map_emo2(hidden_states_emo12_832))

        hidden_states12 = torch.cat(
            [hidden_states_cont12, hidden_states_emo12_256, obj_embedding_level12, obj_embedding_person12], dim=2)
        if data["target11"].shape[0] == 1:
            tgt_mask11 = self.biased_mask1[:, :hidden_states11.shape[1], :hidden_states11.shape[1]].clone().detach().to(
                device=self.device)
            tgt_mask22 = self.biased_mask1[:, :hidden_states12.shape[1], :hidden_states12.shape[1]].clone().detach().to(
                device=self.device)

        memory_mask11 = enc_dec_mask(self.device, hidden_states11.shape[1], hidden_states11.shape[1])
        memory_mask12 = enc_dec_mask(self.device, hidden_states12.shape[1], hidden_states12.shape[1])
        bs_out11 = self.transformer_decoder(hidden_states11, hidden_states_emo11_832, tgt_mask=tgt_mask11,
                                            memory_mask=memory_mask11)
        bs_out12 = self.transformer_decoder(hidden_states12, hidden_states_emo12_832, tgt_mask=tgt_mask22,
                                            memory_mask=memory_mask12)
        bs_output11 = self.bs_map_r(bs_out11)
        bs_output12 = self.bs_map_r(bs_out12)

        return bs_output11, bs_output12, label1

    def predict(self, audio, level, person):
        frame_num11 = math.ceil(audio.shape[1] / 16000 * 30)
        # frame_num11 = math.ceil(audio.shape[1] / 16000 * 25)
        # ----------------- change to 25 fps -----------------
        
        
        # content encoder
        inputs12 = self.processor(torch.squeeze(audio), sampling_rate=16000, return_tensors="pt",
                                  padding="longest").input_values.to(self.device)
        hidden_states_cont1 = self.audio_encoder_cont(inputs12, frame_num=frame_num11).last_hidden_state
        
        # emotion encoder
        inputs12 = self.feature_extractor(torch.squeeze(audio), sampling_rate=16000, padding=True,
                                          return_tensors="pt").input_values.to(self.device)
        output_emo1 = self.audio_encoder_emo(inputs12, frame_num=frame_num11)
        hidden_states_emo1 = output_emo1.hidden_states
        
        # ==observe the emotion prediction==
        # inputs = self.feature_extractor(torch.squeeze(audio), sampling_rate=16000, return_tensors="pt", padding="longest")
        with torch.no_grad():
            # outputs = self.audio_encoder_emo(inputs.input_values)
            predictions = torch.nn.functional.softmax(output_emo1.logits.mean(dim=1), dim=-1) # Average over sequence length
            predicted_label = torch.argmax(predictions, dim=-1)  # Get the predicted label
            emotion = self.audio_encoder_emo.config.id2label[predicted_label.item()]  # Convert to emotion label
        print("="*80)
        print(f'Predicted emotion: {emotion}')
        # print id2label and corresponding emotion ids
        print(f'Emotion labels: {self.audio_encoder_emo.config.id2label}')
        print("="*80)
        # ==observe the emotion prediction==

        # onehot level of emotion and person style (get from out input parameter)
        onehot_level = self.one_hot_level[level]
        onehot_level = torch.from_numpy(onehot_level).to(self.device).float()
        onehot_person = self.one_hot_person[person]
        onehot_person = torch.from_numpy(onehot_person).to(self.device).float()
        
        # testing
        print('='*80)
        # print('frame_num11:', frame_num11)
        # print('inputs12:', inputs12)
        # print('inputs12.shape:', inputs12.shape)
        # print('hidden_states_cont1:', hidden_states_cont1)
        # print('hidden_states_cont1.shape:', hidden_states_cont1.shape)
        # print('hidden_states_emo1:', hidden_states_emo1)
        # print('hidden_states_emo1.shape:', hidden_states_emo1.shape)
        # print('onehot_level:', onehot_level)
        # print('onehot_level.shape:', onehot_level.shape)
        # print('onehot_person:', onehot_person)
        # print('onehot_person.shape:', onehot_person.shape)
        
        # print('output_emotion1:', output_emo1)
        # print('output_emotion1.type:', type(output_emo1))
        # print('output_emotion1.shape:', output_emo1.shape)
        
        # print('='*80)
        
        # test output
        # [EmoTalk] ================================================================================
        # [EmoTalk] frame_num11: 120
        # [EmoTalk] inputs12: tensor([[-8.8177e-05, -8.8177e-05, -8.8177e-05,  ..., -8.8177e-05,
        # [EmoTalk] -8.8177e-05, -8.8177e-05]], device='cuda:0')
        # [EmoTalk] inputs12.shape: torch.Size([1, 63829])
        # [EmoTalk] hidden_states_cont1: tensor([[[ 0.0862,  0.0858, -0.0076,  ...,  0.0513, -0.3031, -0.1376],
        # [EmoTalk] [ 0.0057, -0.0317, -0.0961,  ..., -0.0411, -0.2143, -0.1755],
        # [EmoTalk] [ 0.0069, -0.0714, -0.1365,  ..., -0.0460, -0.1821, -0.1546],
        # [EmoTalk] ...,
        # [EmoTalk] [-0.0066,  0.1593,  0.0422,  ..., -0.1917,  0.4865,  0.0873],
        # [EmoTalk] [-0.1065,  0.2869,  0.0926,  ..., -0.1469,  0.2897, -0.0310],
        # [EmoTalk] [-0.3871,  0.0331,  0.1422,  ...,  0.2580, -0.1615,  0.1293]]],
        # [EmoTalk] device='cuda:0')
        # [EmoTalk] hidden_states_cont1.shape: torch.Size([1, 120, 1024])
        
        
        # [EmoTalk] hidden_states_emo1: tensor([[[ 0.0485, -0.1300, -0.0277,  ...,  0.2553,  0.0064, -0.0318],
        # [EmoTalk] [ 0.0478, -0.1221, -0.0266,  ...,  0.2416,  0.0173, -0.0301],
        # [EmoTalk] [ 0.0434, -0.1163, -0.0242,  ...,  0.2168,  0.0298, -0.0242],
        # [EmoTalk] ...,
        # [EmoTalk] [ 0.0346, -0.1520, -0.0264,  ...,  0.2756, -0.0153, -0.0369],
        # [EmoTalk] [ 0.0340, -0.1524, -0.0268,  ...,  0.2852, -0.0217, -0.0376],
        # [EmoTalk] [ 0.0291, -0.1530, -0.0275,  ...,  0.2929, -0.0312, -0.0413]]],
        # [EmoTalk] device='cuda:0')
        
        # [EmoTalk] hidden_states_emo1.shape: torch.Size([1, 120, 1024])
        
        
        # [EmoTalk] onehot_level: tensor([0., 1.], device='cuda:0')
        # [EmoTalk] onehot_level.shape: torch.Size([2])
        # [EmoTalk] onehot_person: tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        # [EmoTalk] 0., 0., 0., 0., 0., 0.], device='cuda:0')
        # [EmoTalk] onehot_person.shape: torch.Size([24])
        # [EmoTalk] ================================================================================
        
        
        
        
        
        if audio.shape[0] == 1:
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0)
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0)
        else:
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0).permute(1, 0, 2)
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0).permute(1, 0, 2)

        obj_embedding_level11 = obj_embedding_level.repeat(1, frame_num11, 1)
        obj_embedding_person11 = obj_embedding_person.repeat(1, frame_num11, 1)
        hidden_states_cont1 = self.audio_feature_map_cont(hidden_states_cont1)
        hidden_states_emo11_832 = self.audio_feature_map_emo(hidden_states_emo1)
        hidden_states_emo11_256 = self.relu(
            self.audio_feature_map_emo2(hidden_states_emo11_832))
        
        # testing
        # print('='*80)
        # print('hidden_states_cont1:', hidden_states_cont1)
        # print('hidden_states_cont1.shape:', hidden_states_cont1.shape)
        # print('hidden_states_emo11_832:', hidden_states_emo11_832)
        # print('hidden_states_emo11_832.shape:', hidden_states_emo11_832.shape)
        # print('hidden_states_emo11_256:', hidden_states_emo11_256)
        # print('hidden_states_emo11_256.shape:', hidden_states_emo11_256.shape)
        # print('obj_embedding_level11:', obj_embedding_level11)
        # print('obj_embedding_level11.shape:', obj_embedding_level11.shape)
        # print('obj_embedding_person11:', obj_embedding_person11)
        # print('obj_embedding_person11.shape:', obj_embedding_person11.shape)
        # print('='*80)
        
        # e = hidden_states_emo11_256.cpu().detach().numpy()
        # np.savetxt('./../emo266_hidden_angry2.csv', e[0], delimiter=',', fmt='%f')
        

        hidden_states11 = torch.cat(
            [hidden_states_cont1, hidden_states_emo11_256, obj_embedding_level11, obj_embedding_person11], dim=2)
        if audio.shape[0] == 1:
            tgt_mask11 = self.biased_mask1[:, :hidden_states11.shape[1],
                         :hidden_states11.shape[1]].clone().detach().to(device=self.device)

        memory_mask11 = enc_dec_mask(self.device, hidden_states11.shape[1], hidden_states11.shape[1])
        bs_out11 = self.transformer_decoder(hidden_states11, hidden_states_emo11_832, tgt_mask=tgt_mask11,
                                            memory_mask=memory_mask11)
        bs_output11 = self.bs_map_r(bs_out11)

        return bs_output11
