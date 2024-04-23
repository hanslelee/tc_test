import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from simple_ntc.trainer import Trainer
from simple_ntc.data_loader import DataLoader

from simple_ntc.models.rnn import RNNClassifier
from simple_ntc.models.cnn import CNNClassifier


def define_argparser():
    '''
    Define argument parser to set hyper-parameters.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True) #모델 파일네임
    p.add_argument('--train_fn', required=True) #학습에 쓸 트레인 파일, 이걸 train:val = 8:2로 쪼개서 쓸 것임.
    
    p.add_argument('--gpu_id', type=int, default=-1) #cpu는 -1, gpu는 0부터
    p.add_argument('--verbose', type=int, default=2) #얼마나 로그 자주 출력할거냐, 0:출력없음, 1: 에폭이 끝날때마다, 2: 이터레이션마다 정보주는걸로 코딩해놓음

    p.add_argument('--min_vocab_freq', type=int, default=5) #최소 5번 이상 나오는 단어들에 대해서만 클래시파이어가 학습을 하자.
    p.add_argument('--max_vocab_size', type=int, default=999999)

    p.add_argument('--batch_size', type=int, default=256) #미니배치 싸이즈
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--word_vec_size', type=int, default= 256)
    p.add_argument('--dropout', type=float, default=.3)

    p.add_argument('--max_length', type=int, default=256)
    
    #rnn 학습 시 파라미터
    p.add_argument('--rnn', action='store_true')
    p.add_argument('--hidden_size', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=4)
    
    #cnn 학습 시 파라미터
    p.add_argument('--cnn', action='store_true')
    p.add_argument('--use_batch_norm', action='store_true')
    p.add_argument('--window_sizes', type=int, nargs='*', default=[3, 4, 5])
    p.add_argument('--n_filters', type=int, nargs='*', default=[100, 100, 100])

    config = p.parse_args()

    return config


def main(config):
    loaders = DataLoader(
        train_fn=config.train_fn,
        batch_size=config.batch_size,
        min_freq=config.min_vocab_freq,
        max_vocab=config.max_vocab_size,
        device=config.gpu_id
    )

    print(
        '|train| =', len(loaders.train_loader.dataset),
        '|valid| =', len(loaders.valid_loader.dataset),
    )
    
    vocab_size = len(loaders.text.vocab)
    n_classes = len(loaders.label.vocab)
    print('|vocab| =', vocab_size, '|classes| =', n_classes)

    #rnn cnn둘다 안넣었으면 에러
    if config.rnn is False and config.cnn is False:
        raise Exception('You need to specify an architecture to train. (--rnn or --cnn)')

    # rnn 파라미터로 받앗으면 rnn 학습
    if config.rnn:
        # Declare model and loss.
        model = RNNClassifier(
            input_size=vocab_size,
            word_vec_size=config.word_vec_size,
            hidden_size=config.hidden_size,
            n_classes=n_classes,
            n_layers=config.n_layers,
            dropout_p=config.dropout,
        )
        optimizer = optim.Adam(model.parameters())
        crit = nn.NLLLoss()
        print(model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        rnn_trainer = Trainer(config)
        rnn_model = rnn_trainer.train(
            model,
            crit,
            optimizer,
            loaders.train_loader,
            loaders.valid_loader
        )
    if config.cnn:
        # Declare model and loss.
        model = CNNClassifier(
            input_size=vocab_size,
            word_vec_size=config.word_vec_size,
            n_classes=n_classes,
            use_batch_norm=config.use_batch_norm,
            dropout_p=config.dropout,
            window_sizes=config.window_sizes,
            n_filters=config.n_filters,
        )
        optimizer = optim.Adam(model.parameters())
        crit = nn.NLLLoss()
        print(model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        cnn_trainer = Trainer(config)
        cnn_model = cnn_trainer.train(
            model,
            crit,
            optimizer,
            loaders.train_loader,
            loaders.valid_loader
        )

    torch.save({
        'rnn': rnn_model.state_dict() if config.rnn else None,
        'cnn': cnn_model.state_dict() if config.cnn else None,
        'config': config,
        'vocab': loaders.text.vocab,
        'classes': loaders.label.vocab,
    }, config.model_fn) # config.model_fn 파일 이름으로 저장


if __name__ == '__main__':
    config = define_argparser()
    main(config)
