import argparse

from args import train_argparser, eval_argparser
from config_reader import process_configs
from diffusent import input_reader
from diffusent.diffusent_trainer import DiffuSentTrainer
import warnings

warnings.filterwarnings("ignore")


def __train(run_args):
    trainer = DiffuSentTrainer(run_args)
    trainer.train()


def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)


def __eval(run_args):
    trainer = DiffuSentTrainer(run_args)
    trainer.eval()


def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()

    if args.mode == 'train':
        _train()
    elif args.mode == 'eval':
        _eval()
    else:
        raise Exception("Mode not in ['train', 'eval'], e.g. 'python diffusent.py train'")
