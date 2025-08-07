ㄱimport torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    🔑 iTransformer 핵심 모델: 기존 Transformer의 차원을 뒤바꾼 구조
    Paper link: https://arxiv.org/abs/2310.06625
    
    핵심 아이디어: 시간 토큰 → 변수 토큰으로 변환
    - 기존: (B, L, N) → L개의 시간 토큰으로 처리
    - iTransformer: (B, L, N) → N개의 변수 토큰으로 처리
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len      # 입력 시퀀스 길이 (예: 96)
        self.pred_len = configs.pred_len    # 예측 시퀀스 길이 (예: 96, 192, 336, 720)
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm    # 정규화 사용 여부
        
        # 🔄 핵심! DataEmbedding_inverted - 여기서 차원이 뒤바뀜
        # 일반 Transformer: (B,L,N) → (B,L,d_model)
        # iTransformer: (B,L,N) → (B,N,d_model) 
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        
        # 🏗️ Encoder-only 구조 (Decoder 없음)
        # N개의 변수 토큰들이 서로 attention하여 상관관계 학습
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        # 🎯 FullAttention: 변수 간 완전한 attention 계산
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)  # e_layers만큼 쌓음 (예: 3층)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # 📈 Projector: d_model → pred_len으로 변환 (최종 예측 생성)
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        📊 iTransformer의 예측 과정 (핵심 forward pass)
        
        입력: x_enc (B, L, N) - Batch × 시간길이 × 변수개수
        출력: dec_out (B, S, N) - Batch × 예측길이 × 변수개수
        """
        
        # 📏 1단계: 정규화 (Non-stationary Transformer 기법)
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()    # 각 변수별 평균 계산
            x_enc = x_enc - means                           # 평균 0으로 만들기
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # 표준편차 계산
            x_enc /= stdev                                  # 표준편차 1로 정규화

        _, _, N = x_enc.shape # B L N
        # B: batch_size    E: d_model(512)
        # L: seq_len(96)   S: pred_len(96,192,336,720) 
        # N: 변수 개수(ETTh1=7) ← 🔑 이것이 토큰이 됨!

        # 🔄 2단계: Inverted Embedding (핵심!)
        # 일반 Transformer: (B,L,N) → (B,L,d_model) - L개 시간 토큰
        # iTransformer: (B,L,N) → (B,N,d_model) - N개 변수 토큰
        enc_out = self.enc_embedding(x_enc, x_mark_enc) 
        # 내부에서 x.permute(0,2,1) 해서 (B,L,N) → (B,N,L) → Linear → (B,N,d_model)
        
        # 🎯 3단계: 변수 간 Attention (가장 중요한 부분!)
        # (B,N,d_model) → (B,N,d_model) 
        # N×N attention matrix로 변수 간 상관관계 학습
        # 예: ETTh1에서 7×7 attention으로 7개 변수가 서로 얼마나 관련있는지 학습
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 📈 4단계: 예측값 생성 및 차원 복원
        # (B,N,d_model) → (B,N,pred_len) → (B,pred_len,N)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # covariates 제거

        # 📏 5단계: 역정규화 (원래 스케일로 복원)
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))  # 표준편차 복원
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))   # 평균 복원

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        🚀 메인 forward 함수 (훈련/추론 시 호출)
        
        x_enc: 입력 시계열 (B, L, N)
        x_mark_enc: 시간 특성 (월, 일, 시간 등)
        x_dec, x_mark_dec: iTransformer에서는 사용하지 않음 (Encoder-only)
        """
        # forecast 함수에서 실제 예측 수행
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # 최종 예측 길이만큼만 잘라서 반환
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns  # attention weights도 함께 반환
        else:
            return dec_out[:, -self.pred_len:, :]  # 예측값만 반환 [B, pred_len, N]