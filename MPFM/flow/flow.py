# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from MPFM.utils.mask import make_pad_mask
from MPFM.utils.common import f0_to_coarse
from MPFM.flow.modules_grl import SpeakerClassifier
import random
import numpy as np
import pdb

#spk_criterion = nn.CosineEmbeddingLoss()


class MaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 encoder: torch.nn.Module = None,
                 encoder_formant: torch.nn.Module = None,
                 encoder_pitch: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1, 'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine', 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}), 'decoder_params': {'hidden_channels':400,'out_channels':100,'filter_channels':512,'p_dropout':0.1,'n_layers':2,'n_heads':2,'kernel_size':3,'gin_channels':100}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050, 'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        #self.pitch_embed_affine_layer = torch.nn.Linear(1, output_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.adapter_layer=torch.nn.Linear(256, output_size) # 256: enc_formant.output + enc_pit.output
        self.pit_embedding = nn.Embedding(256, output_size)
        self.encoder = encoder
        self.encoder_formant = encoder_formant
        self.encoder_pitch = encoder_pitch
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size()+1, output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss
        self.mel_mask_emb = torch.nn.Embedding(2, 100)
        #self.speaker_classifier = SpeakerClassifier(self.encoder.output_size(),spk_embed_dim)
        with open('data/bad_token.txt') as f:
            lines = f.readlines()
        self.bad_tokens = []
        #self.bad_tokens_weights = []
        for i in lines:
            self.bad_tokens.append(int(i.split(',')[0]))
            #self.bad_tokens_weights.append(-float(i.strip().split(',')[1]))
        self.bad_tokens_weights = np.arange(len(self.bad_tokens),0,-1)

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)
        pitch = batch['pitch'].to(device)
        enr = batch['enr'].to(device)
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        num_replace = [int(0.3 * t_len) for t_len in token_len]
        for i in range(len(token)):
            random_bad_tokens= torch.tensor(random.choices(self.bad_tokens,weights=self.bad_tokens_weights,k=num_replace[i]),dtype=torch.int32).to(token.device)
            random_positions = torch.tensor(random.sample(range(token_len[i]), num_replace[i]),dtype=torch.long).to(token.device)
            #pdb.set_trace()
            token[torch.LongTensor([i]).unsqueeze(1), random_positions] =random_bad_tokens

        #pitch_embedding = self.pitch_embed_affine_layer(pitch.unsqueeze(1).transpose(2, 1))
        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0).long()) * mask

        # text encode
        h_encout, h_lengths = self.encoder(token, token_len)
        #spk_preds = self.speaker_classifier(h_encout.transpose(1,2))
        #h = self.encoder_proj(h_encout)

        h = self.encoder_proj(torch.concat([h_encout,enr.unsqueeze(-1)], dim=2))
        h, h_lengths = self.length_regulator(h, feat_len)

        f0 = f0_to_coarse(pitch)
        f0 = self.pit_embedding(f0)
        
        h_formant, _ = self.encoder_formant(h+embedding.unsqueeze(1), h_lengths)
        h_pitch, _ = self.encoder_pitch(torch.concat([h+embedding.unsqueeze(1),f0], dim=2), h_lengths)

        h_final = self.adapter_layer(torch.concat([h_formant, h_pitch], dim=2))

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        mel_mask = torch.zeros((h.size(0),h.size(1))).to(h)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index=random.randint(0, int(0.3*j))
            conds[i, : index] = feat[i, :index]
            mel_mask[i,:index] = 1
        mel_mask_emb = self.mel_mask_emb(mel_mask.long())
        h_final = h_final + mel_mask_emb
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(feat_len)).to(h)

        feat = F.interpolate(feat.unsqueeze(dim=1), size=h.shape[1:], mode="nearest").squeeze(dim=1)

        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h_final.transpose(1, 2).contiguous(),
            embedding,
            cond=conds
        )
        #spk_loss = spk_criterion(batch['embedding'].to(device), spk_preds, torch.Tensor(spk_preds.size(0)).fill_(1.0).to(device))
        #loss_all = spk_loss*0.5+loss
        #return {'loss': loss_all, 'fm': loss, 'spk':spk_loss}
        return {'loss': loss, 'fm': loss, 'spk':loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  embedding,
                  pitch,
                  enr,
                  prompt_token=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_token_len=torch.zeros(1, dtype=torch.int32),
                  prompt_feat=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_feat_len=torch.zeros(1, dtype=torch.int32),
                  prompt_pit=torch.zeros(1, dtype=torch.int32),
                  prompt_enr=torch.zeros(1, dtype=torch.int32),
                  ):
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)


        # concat text and prompt_text
        token, token_len = torch.concat([token, prompt_token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        enr = torch.concat([enr, prompt_enr], dim=1)
        h = self.encoder_proj(torch.concat([h,enr.unsqueeze(-1)], dim=2))
        #h = self.encoder_proj(h)
        #feat_len = (token_len / 50 * 32000 / 320).int()
        feat_len = token_len *2
        h, h_lengths = self.length_regulator(h, feat_len)

        #feat_len_prompt = (prompt_token_len/ 50 * 32000 / 320).int()
        feat_len_prompt = prompt_token_len * 2
        pitch=pitch[:,:feat_len-feat_len_prompt]
        f0 = f0_to_coarse(pitch)
        f0 = self.pit_embedding(f0)

        prompt_pit=prompt_pit[:,:feat_len_prompt]
        prompt_f0 = f0_to_coarse(prompt_pit)
        prompt_f0 = self.pit_embedding(prompt_f0)
        f0 = torch.concat([f0, prompt_f0], dim=1)
        
        h_formant, _ = self.encoder_formant(h+embedding.unsqueeze(1), h_lengths)
        h_pitch, _ = self.encoder_pitch(torch.concat([h+embedding.unsqueeze(1),f0], dim=2), h_lengths)
        h_final = self.adapter_layer(torch.concat([h_formant, h_pitch], dim=2))

        mel_mask = torch.zeros((1,feat_len)).to(h)
        mel_mask[0,-feat_len_prompt:] = 1
        mel_mask_emb = self.mel_mask_emb(mel_mask.long())
        h_final = h_final + mel_mask_emb

        #h = h+f0

        # get conditions
        conds = torch.zeros([1, feat_len.max().item(), self.output_size], device=token.device)
        #conds[0, :prompt_feat.size(1)] = prompt_feat[0]
        conds[0, -prompt_feat.size(1):] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(feat_len)).to(h)


        feat = self.decoder(
            mu=h_final.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=30,
        )
        if prompt_feat.shape[1] != 0:
            feat = feat[:, :, :-prompt_feat.shape[1]]
        return feat

    @torch.inference_mode()
    def inference_wo_prompt(self,
                  token,
                  token_len,
                  embedding,
                  pitch,
                  enr,
                  prompt_token=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_token_len=torch.zeros(1, dtype=torch.int32),
                  prompt_feat=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_feat_len=torch.zeros(1, dtype=torch.int32),
                  prompt_pit=torch.zeros(1, dtype=torch.int32),
                  prompt_enr=torch.zeros(1, dtype=torch.int32),
                  ):
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)


        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        #enr = torch.concat([enr, prompt_enr], dim=1)
        h = self.encoder_proj(torch.concat([h,enr.unsqueeze(-1)], dim=2))
        #h = self.encoder_proj(h)
        feat_len = (token_len / 50 * 32000 / 320).int()
        h, h_lengths = self.length_regulator(h, feat_len)

        pitch=pitch[:,:feat_len]
        f0 = f0_to_coarse(pitch)
        f0 = self.pit_embedding(f0)

        h_formant, _ = self.encoder_formant(h+embedding.unsqueeze(1), h_lengths)
        h_pitch, _ = self.encoder_pitch(torch.concat([h+embedding.unsqueeze(1),f0], dim=2), h_lengths)

        h_final = self.adapter_layer(torch.concat([h_formant, h_pitch], dim=2))

        #h = h+f0

        # get conditions
        conds = torch.zeros([1, feat_len.max().item(), self.output_size], device=token.device)
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(feat_len)).to(h)

        feat = self.decoder(
            mu=h_final.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
        )
        return feat
