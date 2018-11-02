import preprocess as prep
from const import *

import torch
import torch.nn as nn

softmax = nn.Softmax(dim=1)


def get_top_n_idx(tensor1d, n):
    d = {}
    for i, x in enumerate(tensor1d.data):
        d[i] = float(x)
    result = {}
    num = 0
    for key, value in reversed(sorted(d.items(), key=lambda i: (i[1], i[0]))):
        result[key] = value
        num += 1
        if num == n:
            break
    return list(result.keys())


def get_word_embed_matrix(sentence, vocab, _pre_trained_embedding):
    with torch.no_grad():
        test_in = prep.tensorFromSentence(vocab, sentence)

        x = _pre_trained_embedding[test_in[0]].view(1, -1)
        for i in test_in[1:]:
            x = torch.cat((x, _pre_trained_embedding[i].view(1, -1)), 0)

        return x


def get_word_embed_avg(vocab, sentence, pre_trained_embedding):
    we_matrix = get_word_embed_matrix(sentence, vocab, pre_trained_embedding)
    return torch.mean(we_matrix, 0)


def get_word_embed_avg_sa(vocab, sentence, pre_trained_embedding):
    we_matrix = get_word_embed_matrix(sentence, vocab, pre_trained_embedding)

    x = we_matrix
    attn_matrix = torch.matmul(x, x.transpose(0, 1))
    # result = attn_matrix
    result = softmax(attn_matrix)
    self_attn_matrix = torch.matmul(result, x)

    # represent sentece by averaging matrix
    applied_sent = torch.mean(self_attn_matrix, 0)
    return applied_sent


def get_hiddens(encoder, decoder, sentence, vocab, batch_size, max_length=MAX_LENGTH):
    """ return hidden vectors h_ba, h_tilda (same notation as Luong et al. (2015) """
    with torch.no_grad():
        input_tensor = prep.tensorFromSentenceBatchWithPadding(vocab, [sentence])

        # because of batch, need expansion for input tensor
        temp = input_tensor
        for _ in range(batch_size-1):
            temp = torch.cat((temp, input_tensor), 0)
        input_tensor = temp

        input_tensor = input_tensor.transpose(0, 1)

        encoder_hidden = encoder.init_hidden(batch_size)
        encoder_outputs = torch.zeros(max_length, batch_size, encoder.hidden_size, device=device)
        encoder_h_bar = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(max_length):
            it = input_tensor[ei].view(batch_size, -1)
            encoder_output, encoder_hidden = encoder(it, encoder_hidden)
            encoder_outputs[ei] = encoder_output.transpose(1, 2).view(batch_size, encoder.hidden_size)
            encoder_h_bar[ei] = encoder_hidden[0][0]

        decoder_input = torch.tensor([batch_size * [SOS_token]], device=device).view(batch_size, 1)  # SOS
        decoder_hidden = encoder_hidden
        decoder_h_tilda = torch.zeros(max_length, decoder.hidden_size, device=device)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention, h_tilda = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.squeeze().view(1, batch_size)
            decoder_h_tilda[di] = h_tilda[0]

        return encoder_h_bar, decoder_h_tilda


def hidden_avg_concat_attn(h_bar, h_tilda, n=-1):
    """
    2018.11.02
    h_bar: RNN encoder t-step hidden
    h_tilda: RNN decoder t-step hidden
    h_dot: weighted sum of h_tilda wrt h_bar
    same notation as Luong et al (2015)

    question representation: [h_bar:h_dot]
    answer representation: [h_tilda]
    :return: question_embed, answer_embed
    """
    # TODO
    associate_matrix = torch.matmul(h_bar, h_tilda.transpose(0, 1))
    reduced = torch.sum(associate_matrix, 0)
    high_rel_idx = get_top_n_idx(reduced, n)

    if n > 0:
        # if n is specified make new matrix for using only top n words
        high_rel_answer = h_tilda[high_rel_idx[0]].unsqueeze(0)
        for i in range(1, len(high_rel_idx)):
            high_rel_answer = torch.cat((high_rel_answer, h_tilda[high_rel_idx[i]].unsqueeze(0)), 0)

        associate_matrix = torch.matmul(h_bar, high_rel_answer.transpose(0, 1))

    a_matrix = softmax(associate_matrix, dim=1)
    return torch.matmul(a_matrix, h_tilda.transpose(0, 1))


def get_qa_embed(encoder, decoder, sent, vocab, batch_size):
    h_bar, h_tilda = get_hiddens(encoder, decoder, sent, vocab, batch_size)
    # h_dot = weighted_sum_h_tilda(h_bar, h_tilda, n)
    new_h_bar = torch.cat((h_bar, h_tilda), dim=0)
    # h_bar.size() = (15, 300)
    q_embed = torch.mean(new_h_bar, 0)
    a_embed = torch.mean(h_tilda, 0)
    return q_embed, a_embed

