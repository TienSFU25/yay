import torch
import pickle
import random
import os
import pdb
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

USE_CUDA = torch.cuda.is_available()
flatten = lambda l: [item for sublist in l for item in sublist]

EOS = '<EOS>'
UNK = '<UNK>'
PAD = '<PAD>'
SOS = '<SOS>'

def pad(some_sequence, length=60):
    if len(some_sequence) < length:
        while len(some_sequence) < length:
            some_sequence.append(PAD)
    else:
        some_sequence = some_sequence[:length]
        some_sequence[-1] = EOS

    return some_sequence

def _prepare_sequence(seq, to_ix):
    idxs = list(map(lambda w: to_ix[w] if w in to_ix.keys() else to_ix["<UNK>"], seq))
    tensor = Variable(torch.LongTensor(idxs))

    tensor = tensor.view(1, -1)

    return tensor

def prepare_sequence(seq, labenc):
    # idxs = list(map(lambda w: to_ix[w] if w in to_ix.keys() else to_ix["<UNK>"], seq))
    idxs = labenc.transform(seq)
    tensor = Variable(torch.LongTensor(idxs))

    tensor = tensor.view(1, -1)

    return tensor

def preprocessing(file_path, length):
    """
    atis-2.train.w-intent.iob
    """
    processed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/")
    print("processed_data_path : %s" % processed_path)

    if os.path.exists(os.path.join(processed_path, "processed_train_data.pkl")):
        train_data, word2index, tag2index, intent2index = pickle.load(open(os.path.join(processed_path,
                                                                                        "processed_train_data.pkl"),
                                                                           "rb"))
        return train_data, word2index, tag2index, intent2index
                                  
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    
    try:
        train = open(file_path, "r").readlines()
        print("Successfully load data. # of set : %d " % len(train))
    except:
        print("No such file!")
        return None, None, None, None

    try:
        # orig_train = train
        # train = [t[:-1] for t in train]
        # train = [[t.split("\t")[0].split(" "), t.split("\t")[1].split(" ")[:-1], t.split("\t")[1].split(" ")[-1]] for t in train]
        # train = [[t[0][1:-1], t[1][1:], t[2]] for t in train]

        # seq_in, seq_out, intent = list(zip(*train))

        # sentences = []
        # slots = []
        # intents = []

        vocab = set()
        slot_tag = set()
        intent_tag = set()

        padded_sentences = []
        padded_slots = []
        all_intents = []

        for i in range(len(train)):
            # BOS what's restriction ap68 EOS	O O O B-restriction_code O \n
            bunch_of_crap = train[i]

            # what's restriction ap68 EOS	O O O B-restriction_code O
            line = bunch_of_crap[:-1]

            tab_split = line.split("\t")
            words_in_sentence = tab_split[0].split(" ")[1:-1]
            slots_in_sentence = tab_split[1].split(" ")[:-1][1:]
            intent = tab_split[1].split(" ")[-1]

            assert len(words_in_sentence) == len(slots_in_sentence)
            assert len(intent) > 0

            vocab.update(words_in_sentence)
            slot_tag.update(slots_in_sentence)

            intent_tag.update([intent])

            if len(words_in_sentence) < length:
                words_in_sentence.append(EOS)
            
            padded_sentences.append(pad(words_in_sentence))
            padded_slots.append(pad(slots_in_sentence))
            all_intents.append(intent)

        # vocab = set(flatten(sentences))
        # slot_tag = set(flatten(slots))
        # intent_tag = set(intents)
        print("# of vocab : {vocab}, # of slot_tag : {slot_tag}, # of intent_tag : {intent_tag}"
              .format(vocab=len(vocab), slot_tag=len(slot_tag), intent_tag=len(intent_tag)))

        # sin = []
        # sout = []
        
        # for i in range(len(sentences)):
        #     temp = sentences[i]
        #     if len(temp) < length:
        #         temp.append('<EOS>')
        #         while len(temp) < length:
        #             temp.append('<PAD>')
        #     else:
        #         temp = temp[:length]
        #         temp[-1] = '<EOS>'
        #     sin.append(temp)

        #     temp = slots[i]
        #     if len(temp) < length:
        #         while len(temp) < length:
        #             temp.append('<PAD>')
        #     else:
        #         temp = temp[:length]
        #         temp[-1] = '<EOS>'
        #     sout.append(temp)
        # pdb.set_trace()

        vocab_labenc = LabelEncoder()
        slot_labenc = LabelEncoder()
        intent_labenc = LabelEncoder()
        vocab_labenc.fit([*vocab, PAD, UNK, SOS, EOS])
        slot_labenc.fit([*slot_tag, SOS, PAD])
        intent_labenc.fit([*intent_tag])
                
        # word2index = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        # for token in vocab:
        #     if token not in word2index.keys():
        #         word2index[token] = len(word2index)

        # tag2index = {'<PAD>': 0}
        # for tag in slot_tag:
        #     if tag not in tag2index.keys():
        #         tag2index[tag] = len(tag2index)

        # intent2index = {}
        # for ii in intent_tag:
        #     if ii not in intent2index.keys():
        #         intent2index[ii] = len(intent2index)
        # pdb.set_trace()
        train = list(zip(padded_sentences, padded_slots, all_intents))
                
        train_data = []

        for tr in train:

            # temp = prepare_sequence(tr[0], word2index)

            # temp2 = prepare_sequence(tr[1], tag2index)

            temp = prepare_sequence(tr[0], vocab_labenc)

            temp2 = prepare_sequence(tr[1], slot_labenc)

            as_idx = intent_labenc.transform([tr[2]])
            temp3 = Variable(torch.LongTensor(as_idx))
            # pdb.set_trace()

            train_data.append((temp, temp2, temp3))
        
        # pickle.dump((train_data,word2index,tag2index,intent2index),open(os.path.join(processed_path, "processed_train_data.pkl"), "wb"))
        # pickle
        print("Preprocessing complete!")
        # return train_data, word2index, tag2index, intent2index
        return train_data, vocab_labenc, slot_labenc, intent_labenc

    except Exception as e:
        print(e)
        return None, None, None, None              
              
def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex+batch_size
        sindex = temp
        
        yield batch


# def load_dictionary():
    
#     processed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data/")

#     if os.path.exists(os.path.join(processed_path, "processed_train_data.pkl")):
#         _, word2index, tag2index, intent2index \
#             = pickle.load(open(os.path.join(processed_path, "processed_train_data.pkl"), "rb"))
#         return word2index, tag2index, intent2index
#     else:
#         print("Please, preprocess data first")
#         return None, None, None
