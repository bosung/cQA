import argparse
import sent_eval as se
import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='debug')
    parser.add_argument('--encoder', help='load exisited model')
    parser.add_argument('--decoder', help='load exisited model')
    parser.add_argument('--optim', default='RMSprop')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--w_embed_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--save', choices=['y', 'n'], default='n')
    parser.add_argument('--pre_trained_embed', choices=['y', 'n'], default='y')
    args = parser.parse_args()

    if args.mode == 'sent-eval':
        # sentence embedding experiments
        se.evaluate(args)
    elif args.mode == 'train':
        # train model and evaluate accuracy
        train.main(args)
