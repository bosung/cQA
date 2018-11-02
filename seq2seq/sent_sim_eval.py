import sent_emb_model as se
import preprocess as prep
from preprocess import Vocab

from model import *
from const import *
from utils import *

EMBED_PATH = 'data/komoran_hd_2times.vec'
train_file = 'data/train_data_nv.txt'


def eval_sim_lc(encoder, vocab, batch_size, pre_trained_embedding, decoder=None, alpha=0.9):
    train_data, _, test_data, test_answer = prep.prepare_evaluate()

    # embed candidates
    train_q_embed = {}
    train_a_embed = {}
    print("[INFO] encoding train %d data ..." % len(train_data))
    for d in train_data.keys():
        # TODO
        h_bar, h_tilda = se.get_hiddens(encoder, decoder, train_data[d].q, vocab, batch_size)
        # h_dot = weighted_sum_h_tilda(h_bar, h_tilda, n)
        new_h_bar = torch.concat(h_bar, h_tilda, dim=0)
        # train_a_embed[d] = get_hirel_n_ans_avg(h_bar, h_tilda, 3)
        # h_bar.size() = (15, 300)
        train_q_embed[d] = torch.mean(new_h_bar, 0)
        train_a_embed[d] = torch.mean(h_tilda, 0)
    print("[INFO] done")

    print("[INFO] start evaluating!")
    print("==================>", alpha)
    total = len(test_data)
    answer5 = 0
    answer1 = 0
    for i, tk in enumerate(test_data.keys()):
        h_bar_, h_tilda_ = se.get_hiddens(encoder, decoder, test_data[tk].q, vocab, batch_size)

        # a_embed = get_hirel_n_ans_avg(h_bar_, h_tilda_, 3)
        a_embed = torch.mean(h_tilda_, 0)
        q_embed = torch.mean(h_bar_, 0)

        # cacluate score
        temp_q = {}
        temp_a = {}
        temp = {}
        for candi in train_q_embed.keys():
            # question part
            tq = train_q_embed[candi].view(-1)
            eq = q_embed.view(-1)
            temp_q[candi] = cosine_similarity(tq, eq)
            # answer part
            ta = train_a_embed[candi].view(-1)
            ea = a_embed.view(-1)
            temp_a[candi] = cosine_similarity(ta, ea)

            temp[candi] = alpha*temp_q[candi] + (1-alpha)*temp_a[candi]
            # temp[candi] = temp_q[candi]

        # sort by cos_sim
        top_n = get_top_n(temp, 5)
        for e in top_n.keys():
            if isAnswer(e, test_answer[tk]):
                answer5 += 1
                break
        top1 = list(top_n.keys())[0]
        if isAnswer(top1, test_answer[tk]):
            answer1 += 1

    accuracy_at_5 = answer5/total*100
    accuracy_at_1 = answer1/total*100

    print("total: %d, accuracy@5: %.4f, accuracy@1: %.4f" % (total, accuracy_at_5, accuracy_at_1))


def evaluate(args):
    vocab = Vocab()
    vocab.build(train_file)

    batch_size = args.batch_size
    hidden_size = args.hidden_size
    w_embed_size = args.w_embed_size

    if args.pre_trained_embed == 'n':
        encoder = Encoder(vocab.n_words, w_embed_size, hidden_size, batch_size).to(device)
        decoder = AttentionDecoder(vocab.n_words, w_embed_size, hidden_size, batch_size).to(device)
        # decoder = Decoder(vocab.n_words, w_embed_size, hidden_size, batch_size).to(device)
    else:
        # load pre-trained embedding
        weight = vocab.load_weight(path="data/komoran_hd_2times.vec")
        encoder = Encoder(vocab.n_words, w_embed_size, hidden_size, batch_size, weight).to(device)
        decoder = AttentionDecoder(vocab.n_words, w_embed_size, hidden_size, batch_size, weight).to(device)
        # decoder = Decoder(vocab.n_words, w_embed_size, hidden_size, batch_size, weight).to(device)

    if args.encoder:
        encoder.load_state_dict(torch.load(args.encoder))
        print("[INFO] load encoder with %s" % args.encoder)
    if args.decoder:
        decoder.load_state_dict(torch.load(args.decoder))
        print("[INFO] load decoder with %s" % args.decoder)

    # evaluate_similarity(encoder, vocab, batch_size, decoder=decoder)

    pre_trained_embedding = vocab.load_weight(EMBED_PATH)
    eval_sim_lc(encoder, vocab, batch_size, pre_trained_embedding, decoder=decoder, alpha=1)
    eval_sim_lc(encoder, vocab, batch_size, pre_trained_embedding, decoder=decoder, alpha=0.95)
    eval_sim_lc(encoder, vocab, batch_size, pre_trained_embedding, decoder=decoder, alpha=0.9)
    eval_sim_lc(encoder, vocab, batch_size, pre_trained_embedding, decoder=decoder, alpha=0.85)
    eval_sim_lc(encoder, vocab, batch_size, pre_trained_embedding, decoder=decoder, alpha=0.8)
    eval_sim_lc(encoder, vocab, batch_size, pre_trained_embedding, decoder=decoder, alpha=0.7)