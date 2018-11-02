import argparse
import sent_eval as se


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='debug')
    parser.add_argument('--encoder', help='load exisiting model')
    parser.add_argument('--decoder', help='load exisiting model')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--w_embed_size', type=int, default=64)
    parser.add_argument('--pre_trained_embed', choices=['y', 'n'], default='n')
    args = parser.parse_args()

    if args.mode == 'sent_eval':
        se.evaluate(args)
