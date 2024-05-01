import sys
import argparse

import torch
import torch.nn as nn

import time

import torchtext
version = list(map(int, torchtext.__version__.split('.')))
if version[0] <= 0 and version[1] < 9:
    from torchtext import data
else:
    from torchtext.legacy import data

from simple_ntc.models.rnn import RNNClassifier
from simple_ntc.models.cnn import CNNClassifier

import mlflow

from model_results.get_accuracy import calculate_accuracy
from model_results.get_f1_score import calculate_f1_score


def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--output_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--top_k', type=int, default=1)
    p.add_argument('--max_length', type=int, default=256)
    
    p.add_argument('--drop_rnn', action='store_true')
    p.add_argument('--drop_cnn', action='store_true')

    config = p.parse_args()

    return config


def read_text(max_length=256):
    '''
    Read text from standard input for inference.
    '''
    lines = []

    # for line in sys.stdin:
    #     if line.strip() != '':
    #         lines += [line.strip().split(' ')[:max_length]]

    f = open("./data/240501/test_input.txt", encoding='UTF-8')
    lines = f.readlines()

    return lines


def define_field():
    '''
    To avoid use DataLoader class, just declare dummy fields. 
    With those fields, we can retore mapping table between words and indice.
    '''
    return (
        data.Field(
            use_vocab=True,
            batch_first=True,
            include_lengths=False,
        ),
        data.Field(
            sequential=False,
            use_vocab=True,
            unk_token=None,
        )
    )


def main(config):
    mlflow.start_run()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    train_config = saved_data['config']
    rnn_best = saved_data['rnn']
    cnn_best = saved_data['cnn']
    vocab = saved_data['vocab']
    classes = saved_data['classes']

    vocab_size = len(vocab)
    n_classes = len(classes)

    text_field, label_field = define_field()
    text_field.vocab = vocab
    label_field.vocab = classes

    lines = read_text(max_length=config.max_length)

    ## 시작 시간
    start_time = time.time()
    print("Start Time:", start_time)

    with torch.no_grad():
        ensemble = []
        if rnn_best is not None and not config.drop_rnn:
            # Declare model and load pre-trained weights.
            model = RNNClassifier(
                input_size=vocab_size,
                word_vec_size=train_config.word_vec_size,
                hidden_size=train_config.hidden_size,
                n_classes=n_classes,
                n_layers=train_config.n_layers,
                dropout_p=train_config.dropout,
            )
            model.load_state_dict(rnn_best)
            ensemble += [model]
        if cnn_best is not None and not config.drop_cnn:
            # Declare model and load pre-trained weights.
            model = CNNClassifier(
                input_size=vocab_size,
                word_vec_size=train_config.word_vec_size,
                n_classes=n_classes,
                use_batch_norm=train_config.use_batch_norm,
                dropout_p=train_config.dropout,
                window_sizes=train_config.window_sizes,
                n_filters=train_config.n_filters,
            )
            model.load_state_dict(cnn_best)
            ensemble += [model]

        y_hats = []
        # Get prediction with iteration on ensemble.
        for model in ensemble:
            if config.gpu_id >= 0:
                model.cuda(config.gpu_id)
            # Don't forget turn-on evaluation mode.
            model.eval()

            y_hat = []
            for idx in range(0, len(lines), config.batch_size):                
                # Converts string to list of index.
                x = text_field.numericalize(
                    text_field.pad(lines[idx:idx + config.batch_size]),
                    device='cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu',
                )

                y_hat += [model(x).cpu()]
            # Concatenate the mini-batch wise result
            y_hat = torch.cat(y_hat, dim=0)
            # |y_hat| = (len(lines), n_classes)

            y_hats += [y_hat]

            model.cpu()
        # Merge to one tensor for ensemble result and make probability from log-prob.
        y_hats = torch.stack(y_hats).exp()
        # |y_hats| = (len(ensemble), len(lines), n_classes)
        y_hats = y_hats.sum(dim=0) / len(ensemble) # Get average
        # |y_hats| = (len(lines), n_classes)

        probs, indice = y_hats.topk(config.top_k)

        ## 종료 시간
        end_time = time.time()
        print("End Time:", end_time)
        
        execution_time = end_time - start_time
        print("Execution Time:", execution_time)

        # rnn_output_file = './output/rnn_test_output_bs256_epoch20.txt'
        # cnn_output_file = './output/cnn_test_output_bs256.txt'
        with open(config.output_fn, 'w', encoding='utf-8') as f:
            for i in range(len(lines)):
                # sys.stdout.write('%s\t%s\n' % (
                #     ' '.join([classes.itos[indice[i][j]] for j in range(config.top_k)]), 
                #     ' '.join(lines[i])
                # ))
                f.writelines('%s\t%s' % (
                    ''.join([classes.itos[indice[i][j]] for j in range(config.top_k)]), 
                    ''.join(lines[i])
                ))
        # f.close()

        answer_file = './data/240501/test_answer.txt'
        model_accuracy = calculate_accuracy(answer_file, config.output_fn)
        true_positives, false_positives, false_negatives, precision, recall, f1_score = calculate_f1_score(answer_file, config.output_fn)

        # 추론 결과 기록
        mlflow.log_param('batch_size', config.batch_size)
        mlflow.log_param('top_k', config.top_k)
        mlflow.log_param('max_length', config.max_length)
        mlflow.log_param('drop_rnn', config.drop_rnn)
        mlflow.log_param('drop_cnn', config.drop_cnn)
        mlflow.log_metric('execution_time', execution_time)
        mlflow.log_metric('model_accuracy', model_accuracy)
        mlflow.log_metric('true_positives', true_positives)
        mlflow.log_metric('false_positives', false_positives)
        mlflow.log_metric('false_negatives', false_negatives)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1_score)

        # 추론 결과 파일 기록
        mlflow.log_artifact(config.output_fn, 'rnn_output')
        # mlflow.log_artifact(cnn_output_file, 'cnn_output')

        mlflow.end_run()


if __name__ == '__main__':
    config = define_argparser()
    main(config)
