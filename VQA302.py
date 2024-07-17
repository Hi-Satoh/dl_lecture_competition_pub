#######################################################################
#
#   DL基礎講座2024　最終課題「Visual Question Answering（VQA）」 VQA302
#
#   自然言語処理　　Tokenizer ＋ 独自辞書
#   モデル        Image（VGG19) Text (LSTM) VQAシステム（Encoder)
#
#######################################################################

##########################################
#       1 環境設定
#########################################
from google.colab import drive
drive.mount('/content/drive')
%cd '/content/drive/My Drive/Colab Notebooks/DL基礎講座2024/最終課題/VQA/'

#######################################
#    1-2 ランタイムにデータを作成する
#######################################
#ランタイムライブラリーにデータを移動、unzip
#解凍用
import zipfile

extracted_folder = '/content/data/'
zip_file_path = '/content/drive/MyDrive/Colab Notebooks/DL基礎講座2024/最終課題/VQA/data/train.zip'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
  zip_ref.extractall(extracted_folder)

#解凍用
import zipfile

extracted_folder = '/content/data/'
zip_file_path = '/content/drive/MyDrive/Colab Notebooks/DL基礎講座2024/最終課題/VQA/data/valid.zip'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
  zip_ref.extractall(extracted_folder)


!pip install tqdm
#ライブラリ関連
import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn

import torch.nn.functional  as F

import torchvision
from torchvision import transforms

#プリトレーニングの利用
import torchvision.models as models

from tqdm import tqdm
import time

#トークナイザーによる変更
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

#トークナイザー化で文字列のパディングを行う
from torch.nn.utils.rnn import pad_sequence


#######################################################################
#  2. 辞書作成　システム
# 　質問辞書　　questionに含まれる単語単位の辞書
# 　  ソース　　train.json/valid.jsonの"question"
# 　回答辞書　　answerに含まれるパラグラフの辞書"_"で結合
# 　  ソース　　train.jsonの"answer" 10名の回答をそれぞれパラグラフでトークン化
#
#  質問内容に新しい質問が発生する可能性があるため、追加を入れて作成する。(question)
#  answer 新しい回答の追加についてはやや疑問　回答の使い回し以外難しい　クラス分類になる
#       　商品名はあり得る（学習データが必要）　"what is this"  ラベル名
#######################################################################

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)


def collate_fn(batch):
    images, questions, answers, mode_answers = zip(*batch)

    images = torch.stack(images)
    questions = torch.tensor(questions)
    answers = torch.tensor(answers)
    mode_answers = torch.tensor(mode_answers)

    return images, questions_padded, answers_padded, mode_answers


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_with_phrases(text):
  text =  text.replace(" ", "_")
  return text

def tokenize_to_words(text):
  text =  text.replace("_", " ")
  return text

#トークナイズするためpaddingを考慮してデータの長さを求める
def get_max_seq_length(df, column_name, tokenizer):
    max_len = 0
    for text in df[column_name]:
        tokens = tokenizer(process_text(text))
        max_len = max(max_len, len(tokens))
    return max_len

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)



##############################
#                            #
# 2-1. データローダーの作成       #
#                           #
#############################
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, token_dict, tokenizer, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.token_dict = token_dict  # トークン辞書
        self.answer = answer
        self.tokenizer = tokenizer                   ########トークナイザーにする。
        self.token_dict = token_dict                 ########


        self.max_question_length = get_max_seq_length(self.df, "question", tokenizer)  ######　questionの最大トークン数
        print("max_question_length =", self.max_question_length)
        self.max_answer_length  = 10                          #get_max_seq_length(self.df, "answers" , tokenizer)  ######　answerの最大トークン数
        print("max_answer_length =", self.max_answer_length)

        self.max_question_token_count = self.max_question_length
        self.max_answer_count = self.max_answer_length


    def pad_sequence(self, seq, max_length):
        """
        シーケンスをパディングします。
        Parameters
        ----------
        seq : List[int]
            シーケンス
        max_length : int
            パディング後の最大長
        Returns
        -------
        padded_seq : List[int]
            パディングされたシーケンス
        """
        if len(seq) < max_length:
            seq = seq + [0] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
        return seq


    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question :    -> vocab_sizeはトークンに長さになる
            ”トークン化に変更"
        answers : torch.Tensor  (n_answer)
            トークンインデックスになる
        mode_answer_idx : torch.Tensor  (1)
            トークンインデックスになる
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)

        question_words = process_text(self.df["question"][idx])                       ########
        question_tokens = self.tokenizer(question_words)                              #######
        question_indices = [self.token_dict.get_question_token(token) for token in question_tokens]  #######
        question_indices = self.pad_sequence(question_indices, self.max_question_length)  # パディング

        question_len = sum(1 for num in question_indices if num != 0)

        if self.answer:
            answers = []

            for answer in self.df["answers"][idx]:
                tokens = self.tokenizer(tokenize_with_phrases(process_text(answer["answer"])))
                if tokens: # Check if tokens list is not empty
                    answers.append(self.token_dict.get_answer_token(tokens[0]))
            answers = self.pad_sequence(answers, self.max_answer_length)  # パディング
            mode_answer_idx = mode(answers) # 最頻値を取得（正解ラベル）
            return image, torch.tensor(question_indices, dtype=torch.long), torch.tensor(question_len, dtype=torch.long) , torch.tensor(answers, dtype=torch.long), int(mode_answer_idx)
        else:
            return image, torch.tensor(question_indices, dtype=torch.long), torch.tensor(question_len, dtype=torch.long)


    def max_question_token_count(self):
      return self.max_question_length

    def max_answer_count(self):
      return self.max_answer_length

    def __len__(self):
        return len(self.df)


##############################
#                            #
# 2-2. 辞書の作成（クラス化）    #
#                           #
#############################

class TokenDict:
    def __init__(self, question=True,answer=True):
        if question:
            with open('question_vocab.pkl', 'rb') as f:
              self.question_vocab = pickle.load(f)
            with open('reverse_question_vocab.pkl', 'rb') as f:
              self.reverse_question_vocab = pickle.load(f)
        if answer:
            with open('answer_vocab.pkl', 'rb') as f:
              self.answer_vocab = pickle.load(f)
            with open('reverse_answer_vocab.pkl', 'rb') as f:
              self.reverse_answer_vocab = pickle.load(f)
        self.question_vocab_size = len(self.question_vocab)
        self.answer_vocab_size = len(self.answer_vocab)

    def get_question_token(self, index):
        return self.question_vocab[index]

    def get_question_index(self, token):
        return self.reverse_question_vocab[token]

    def get_answer_token(self, index):
        return self.answer_vocab[index]

    def get_answer_index(self, token):
        return self.reverse_answer_vocab[token]

import pickle
##########################
#  2-2-1 作業用辞書の作成   #
##########################

df = pandas.read_json("./data/train.json")  # 画像ファイルのパス，question, answerを持つDataFrame
# question / answerの辞書を作成
question2idx = {}
answer2idx = {}

# 訓練データより構築
# [質問文]に含まれる単語を辞書に追加
cnt_n = 0
add_n = 0
for question in df["question"]:
    question = process_text(question)
    words = question.split(" ")
    #words = yield_tokens(words)
    cnt_n += 1
    for word in words:
        if word not in question2idx:
            question2idx[word] = len(question2idx)
            add_n += 1
print(f"question1 words({add_n})addition({add_n})\n")


# [回答]に含まれる単語を辞書に追加
cnt_n = 0
cnt_2 = 0
add_n = 0
cnt_l = 0
for answers in df["answers"]:
    cnt_2 += 1
    for answer in answers:
        word = answer["answer"]
        word = process_text(word)
        word = tokenize_with_phrases(word)    #2つの単語でできていても１つにしてしまう。"aa_bb"というパラグラフの単語になる。　"aa" "bb"ではない。
        cnt_n += 1
        if word not in answer2idx:
            answer2idx[word] = len(answer2idx)
            add_n += 1
print(f"answer1 line({cnt_2})words({cnt_n})addition({add_n})\n")



# 検証データより構築
#　[質問文]に含まれる単語を辞書に追加
cnt_n = 0
add_n = 0
df = pandas.read_json("./data/valid.json")
for question in df["question"]:
    question = process_text(question)
    words = question.split(" ")
    cnt_n += 1
    for word in words:
        if word not in question2idx:
            question2idx[word] = len(question2idx)
            add_n += 1
print(f"question2 words({add_n})addition({add_n})\n")


#VizWiz　Class_mappingの単語を登録
#　[回答文]に含まれる単語を回答辞書に追加　　（すでに登録済み　単語は入らない　単語として登録する）
# csv fileを　df_vizwiz へ
file_path = '/content/drive/My Drive/Colab Notebooks/DL基礎講座2024/最終課題/VQA/class_map.csv'  # CSVファイルのパスを指定してください
df_vizwiz = pandas.read_csv(file_path)
#df_vizwiz["answer"]   #"answer"　 "class_id"　に分かれている。

cnt_n = 0
add_n = 0
for word in df_vizwiz["answer"]:
    word = process_text(word)
    word = tokenize_with_phrases(word)    #2つの単語でできていても１つにしてしまう。"aa_bb"というパラグラフの単語になる。　"aa" "bb"ではない。
    cnt_n += 1
    if word not in answer2idx:
        answer2idx[word] = len(answer2idx)
        add_n += 1

print(f"answer(VizWiz words({cnt_n})addition({add_n})\n")
print("question direcrory",len(question2idx))
print("answer direcrory",len(answer2idx))


################################
#   2-2-3 辞書作成と保存         #
#　　　　利用環境の定義        　　#
################################

# トークンリストを生成
tokens = list(question2idx.keys())
# トークンリストからイテレータを生成
tokens_iterator = yield_tokens(tokens)
# 新しい辞書の構築
question_vocab = build_vocab_from_iterator(tokens_iterator, specials=["<UNK>"])
question_vocab.set_default_index(question_vocab["<UNK>"])

# トークンリストを生成
tokens = list(answer2idx.keys())
# トークンリストからイテレータを生成
tokens_iterator = yield_tokens(tokens)
# 新しい辞書の構築
answer_vocab = build_vocab_from_iterator(tokens_iterator, specials=["<UNK>"])
answer_vocab.set_default_index(answer_vocab["<UNK>"])

#逆引き辞書の作成
reverse_question_vocab = {index: token for index, token in enumerate(question_vocab.get_itos())}
reverse_answer_vocab = {index: token for index, token in enumerate(answer_vocab.get_itos())}


#独自事象の利用 単語はコード体系はgoogle basic_englishで処理
token_dict = TokenDict()
##トークナイザーは
tokenizer = get_tokenizer("basic_english")


#保存
with open('question_vocab.pkl', 'wb') as f:
    pickle.dump(question_vocab, f)
with open('reverse_question_vocab.pkl', 'wb') as f:
    pickle.dump(reverse_question_vocab, f)
with open('answer_vocab.pkl', 'wb') as f:
    pickle.dump(answer_vocab, f)
with open('reverse_answer_vocab.pkl', 'wb') as f:
    pickle.dump(reverse_answer_vocab, f)

print("len(question_vocab)=",len(question_vocab))
print("len(answer_vocab  )=",len(answer_vocab))

#############################################
#     3.テストデータ準備                       #
#     datasetの準備(訓練用・テスト用)           #
#     dataloader                            #
#     データ件数　Train 19,873   test 4,969   #
#############################################


batch_size = 30       #128 -> 64

# VGG19用の前処理を定義
transform = transforms.Compose([
    #transforms.Resize(256),  # 画像の短い辺を256にリサイズ
    #transforms.CenterCrop(224),  # 224x224にクロップ
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # ピクセル値をテンソルに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
])

Path = "/content/data/"
#Path = "/content/drive/My Drive/Colab Notebooks/DL基礎講座2024/最終課題/VQA/data/"


train_path = Path + "train.json"
valid_path = Path + "valid.json"
image_dir_train_path = Path + "train"
image_dir_valid_path = Path + "valid"


train_dataset = VQADataset(df_path="./data/train.json", image_dir=image_dir_train_path, token_dict=token_dict, tokenizer=tokenizer, transform=transform)
test_dataset = VQADataset(df_path="./data/valid.json", image_dir=image_dir_valid_path, token_dict=token_dict, tokenizer=tokenizer, transform=transform, answer=False)
#test_dataset.update_dict(train_dataset)

#下記は一時的
#rev_test_dataset = VQADataset(df_path="./data/train.json", image_dir=image_dir_train_path, token_dict=token_dict, tokenizer=tokenizer, transform=transform, answer=False)



#Full data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   #!!!!!!!
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


#多すぎるので、少しにする　（動作確認用）
#N = 300  # サンプリングするデータ数
# subset_train_indices = range(0,5000)
# subset_test_indices = range(0,2000)
# subset_train = torch.utils.data.Subset(train_dataset, subset_train_indices)
# subset_test = torch.utils.data.Subset(test_dataset, subset_test_indices)
#rev_test_loader = torch.utils.data.DataLoader(rev_test_dataset, batch_size=1, shuffle=False)
#partial data
# train_loader = torch.utils.data.DataLoader(subset_train, batch_size=batch_size, shuffle=False)   #!!!!!!!
# test_loader = torch.utils.data.DataLoader(subset_test, batch_size=1, shuffle=False)


###########################################
#     4. VQAモデルの定義  　　　　　　　　　　　#
#       画像エンコーダ                      #
#       テキストエンコーダ                   #
#       モデル定義                          #
###########################################


# 4-1. 評価指標の実装
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)

###########################################
#      4-2 画像エンコーダ　クラス・関数　　　　　#
###########################################
class ImgEncoder(nn.Module):

    def __init__(self, embed_dim):   #=image_feature

        super(ImgEncoder, self).__init__()
        self.model = models.vgg19(pretrained=True)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1]) # remove vgg19 last layer
        self.fc = nn.Linear(in_features, embed_dim)
        print("vgg19 encorder")

    def forward(self, image):

        with torch.no_grad():
            img_feature = self.model(image) # (batch, channel, height, width)
        img_feature = self.fc(img_feature)

        l2_norm = F.normalize(img_feature, p=2, dim=1).detach()
        return l2_norm


###########################################
#      4-3 テキスト系　クラス・関数　　　　　　　#
###########################################

########################################
#   パラメータ
#     word_num = 単語辞書の大きさ             voca_size (10000)
#     emb_dim  = 単語列に長さ（入力トークン数） question最大長さ　(57) XXXXXX間違え
#     hid_dim  = LSTMのh                    調整値　現在　　(100)
#     text_feat = VQAに渡す特徴量ベクトル　　　　調整値　現在　 (512)
########################################

class textEncoder_xx(nn.Module):    #最終決定

    def __init__(self, qu_vocab_size, word_embed, hidden_size, num_hidden, qu_feature_size):
        super().__init__()
        print("textEncorder_xx")
        self.emb = nn.Embedding(10000, word_embed)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed, hidden_size, num_hidden, batch_first=True)  # nn.LSTMの使用  注意!!!!!!!　batch first num_hidden = 2のケースを作る。

        self.attention = SelfAttention(hidden_size)       #encoderの中にattentionを入れる
        self.linear = nn.Linear(5*hidden_size, qu_feature_size)
        self.hidden_size = hidden_size

    def forward(self, x,  len_seq, len_seq_max=0, init_state=None):
        h = self.emb(x)                   #inp (batch_size=128, msg_len=65) => (batch_size_128,msg_len=65 word_embed=128)　　　　　(128, 65, 128)
        h = self.tanh(h)                  ##  -1 から　+1  平準化

        if len_seq_max > 0:
            h, (h_n, c_n) = self.lstm(h[:, 0:len_seq_max, :], init_state)
        else:
            h, (h_n, c_n) = self.lstm(h, init_state)

        # LSTMの出力は(batch_size, seq_len, hidden_size)
        # アテンションを適用
        attn_output = self.attention(h)  # (batch_size, seq_len, hidden_size)

        # LSTMの出力は(batch_size, seq_len, hidden_size)
        # 最終タイムステップの隠れ状態とセル状態を取得
        last_h_n = h_n.view(-1, len(x), self.hidden_size)  # (num_layers, batch_size, hidden_size)
        last_c_n = c_n.view(-1, len(x), self.hidden_size)  # (num_layers, batch_size, hidden_size)

       # 最終レイヤーの隠れ状態とセル状態を取り出し、結合
        h = torch.cat((attn_output[:, -1, :], last_h_n[-1], last_c_n[-1], last_h_n[-2], last_c_n[-2]), dim=1)  # (batch_size, 4 * hidden_size)

        y = self.linear(h)
        return y

class textEncoder2(nn.Module):  #最後まで比較検証

    def __init__(self, qu_vocab_size, word_embed, hidden_size, num_hidden, qu_feature_size):
        super().__init__()
        self.emb = nn.Embedding(qu_vocab_size, word_embed)
        self.rnn_layers = nn.ModuleList([nn.LSTM(word_embed if i == 0 else hidden_size, hidden_size, 1, batch_first=True) for i in range(num_hidden)])
        self.linear = nn.Linear(hidden_size, qu_feature_size)
        self.dropout = nn.Dropout(0.2)  # ドロップアウト層を追加
        self.dropout_layers = nn.ModuleList([nn.Dropout(0.2) for _ in range(num_hidden)])  # 各層のドロップアウト層
        self.num_hidden = num_hidden

    def forward(self, x, len_seq_max=0, len_seq=None, init_state=None):
        h = self.emb(x)
        h = self.dropout(h)     #ドロップアウトを採用
        for i in range(self.num_hidden):
            h = self.dropout_layers[i](h)  # ドロップアウトを適用
            if len_seq_max > 0:
            #if (len_seq_max > 0).all(): # colab sugg
                h, _ = self.rnn_layers[i](h[:, 0:len_seq_max, :], init_state)
            else:
                h, _ = self.rnn_layers[i](h, init_state)

        h = h.transpose(0, 1)
        if len_seq is not None:
            h = h[len_seq - 1, list(range(len(x))), :]
        else:
            h = h[-1]

        y = self.linear(h)
        return y

class QuEncoder(nn.Module):      #最後まで比較検証

    def __init__(self, qu_vocab_size, word_embed, hidden_size, num_hidden, qu_feature_size):

        super(QuEncoder, self).__init__()
        #self.word_embedding = nn.Embedding(qu_vocab_size, word_embed)
        self.word_embedding = nn.Embedding(10000, 128)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed, hidden_size, num_hidden) # input_feature, hidden_feature, num_layer
        self.fc = nn.Linear(2*num_hidden*hidden_size, qu_feature_size)

    def forward(self, question):
        #print("enter emb")
        qu_embedding = self.word_embedding(question)                # (batchsize, qu_length=30, word_embed=300)
        nan_count = torch.isnan(qu_embedding).sum()
        if nan_count > 0:
            print("Warning: NaN detected in qu_embedding tensor.")
            print("nan_count",nan_count)
        qu_embedding = self.tanh(qu_embedding)
        qu_embedding = qu_embedding.transpose(0, 1)                 # (qu_length=30, batchsize, word_embed=300)
        _, (hidden, cell) = self.lstm(qu_embedding)                 # (num_layer=2, batchsize, hidden_size=1024)
        qu_feature = torch.cat((hidden, cell), dim=2)               # (num_layer=2, batchsize, 2*hidden_size=1024)
        qu_feature = qu_feature.transpose(0, 1)                     # (batchsize, num_layer=2, 2*hidden_size=1024)
        qu_feature = qu_feature.reshape(qu_feature.size()[0], -1)   # (batchsize, 2*num_layer*hidden_size=2048)
        qu_feature = self.tanh(qu_feature)
        qu_feature = self.fc(qu_feature)                            # (batchsize, qu_feature_size=1024)
        return qu_feature

###########################################
#     4-3  attention　logic               #
###########################################
############################
#  softmax attention       #   #最後まで比較検証
############################
class Attention(nn.Module):
    def __init__(self, feature_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(feature_size, feature_size)

    def forward(self, features):
        # Compute attention scores
        attn_scores = self.attention(features)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)

        # Apply attention weights to features
        attn_applied = attn_weights * features

        return attn_applied

#############################
#   Self attention           #   #最終的に選択
#############################

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(feature_size, feature_size)
        self.key = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)
        self.scale = feature_size ** 0.5  # スケーリング係数

    def forward(self, x, mask=None):
        Q = self.query(x)  # (batch_size, seq_len, feature_size)
        K = self.key(x)  # (batch_size, seq_len, feature_size)
        V = self.value(x)  # (batch_size, seq_len, feature_size)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, seq_len, seq_len)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))  # Apply the mask

        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, seq_len, seq_len)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, seq_len, feature_size)

        return attn_output


######################################
#      4-4 VQA　クラス・関数　　　　　　　#
######################################

class VQAModel(nn.Module):
    def __init__(self, feature_size, qu_vocab_size, ans_vocab_size, word_embed, hidden_size, num_hidden):
        super(VQAModel, self).__init__()

        self.img_encoder = ImgEncoder(feature_size)

        self.qu_encoder = textEncoder_xx(qu_vocab_size, word_embed, hidden_size, num_hidden, feature_size)

        self.img_attention = Attention(feature_size)      #soft max attention
        self.qst_attention = Attention(feature_size)      #soft max attention

        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()
        #try2
        #self.fc1 = nn.Linear(feature_size, ans_vocab_size)
        #self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)
        #try2'
        self.fc1 = nn.Linear(feature_size, feature_size)
        self.fc2 = nn.Linear(feature_size, ans_vocab_size)

    def forward(self, image, question, q_len):
        img_feature = self.img_encoder(image)               # (batchsize, feature_size=1024)
        qst_feature = self.qu_encoder(question, q_len)      # (batchsize, feature_size=1024

# combine feature の検証
        #(1)  　Phase1  Try-2　　　Try-5 は戻り層をもう一層入れる
        combined_feature = img_feature * qst_feature             #(30, 1024)
        combined_feature = self.tanh(combined_feature)

        #(2)    Phase2  Try-3 logcにNG
        # combined_feature = torch.stack((img_feature, qst_feature), dim=1)
        # combined_feature = self.attention(combined_feature)


        #(3)    Phase3  Try-4  tanhは不要だった可能性あり Try-6 tanhを外す
        # img_feature = self.img_attention(img_feature.unsqueeze(1)).squeeze(1)  # (batchsize, feature_size=1024)
        # qst_feature = self.qst_attention(qst_feature.unsqueeze(1)).squeeze(1)  # (batchsize, feature_size=1024)
        # combined_feature = img_feature * qst_feature  # (batchsize, 1024)
        # combined_feature = self.tanh(combined_feature)

        #(4)    Phase4  Try-7
        # img_feature = self.img_attention(img_feature.unsqueeze(1)).squeeze(1)  # (batchsize, feature_size=1024)
        # qst_feature = self.qst_attention(qst_feature.unsqueeze(1)).squeeze(1)  # (batchsize, feature_size=1024)
        # combined_feature = img_feature + qst_feature  # (batchsize, 1024)


        #common logic
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)       # (batchsize, ans_vocab_size=1000)


        #return combined_feature

        combined_feature = self.dropout(combined_feature)      # Try 5
        combined_feature = self.tanh(combined_feature)

        logits = self.fc2(combined_feature)                 # (batchsize, ans_vocab_size=1000)

        return logits

#######################################
#                                     #
#    5. 実行用の関数定義 (訓練・予測)     #
#                                     #
#######################################

# 5-1 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()

    for image, question,  q_len, answers, mode_answer in tqdm(dataloader):

        image, question, q_len, answers, mode_answer = \
           image.to(device), question.to(device), q_len.to(device), answers.to(device).float(), mode_answer.to(device)  #LSTMの時

        #For Debug
        if torch.isnan(image).any() or torch.isinf(image).any():
            print("Warning: NaN or Inf detected in image tensor.")
        if torch.isnan(question).any() or torch.isinf(question).any():
            print("Warning: NaN or Inf detected in question tensor.")

        pred = model(image, question, q_len)

        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device):  #特殊目的　ttrain dataで予測するケース対応
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, q_len, answers, mode_answer in tqdm(dataloader):     #訓練同じ形
        image, question, q_len, answer, mode_answer = \
        image.to(device), question.to(device),  q_len.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question, q_len)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start



#######################################
#######################################
#                                     #
#    5. 訓練・予測の実施                 #
#                                     #
#######################################
#######################################

from tqdm import tqdm_notebook as tqdm
set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# before Try9
# old
# feature_size= 1024      #   VQAモデルへの特徴量
# qu_vocab_size = 10000   #   単語サイズ　　10000
# ans_vocab_size = 40897  #   分類するクラス　1000  40311  ->40897
# word_embed = 128       #   単語の次元数　64 次元  -> 128  -> 64
# hidden_size = 128       #   LSTMのかくれ層次元  100 -> 128
# num_hidden  =  2       #   LSTM 1 層（東DL)   論文は２層
# qu_feature_size = 1024  #   questionの特徴量　画像と同じ

#Try09
# feature_size= 2048      #   VQAモデルへの特徴量
# qu_vocab_size = 10000   #   単語サイズ　　10000
# ans_vocab_size = 40897  #   分類するクラス　1000  40311  ->40897
# word_embed = 256 (128)  #   単語の次元数　64 次元  -> 128  -> 64
# hidden_size = 256       #   LSTMのかくれ層次元  100 -> 128
# num_hidden  =  2       #   LSTM 1 層（東DL)   論文は２層
# qu_feature_size = 2048  #   questionの特徴量　画像と同じ

#try10
# feature_size= 3072      #   VQAモデルへの特徴量
# qu_vocab_size = 10000   #   単語サイズ　　10000
# ans_vocab_size = 40897  #   分類するクラス　1000  40311  ->40897
# word_embed = 256 (128)  #   単語の次元数　64 次元  -> 128  -> 64 ->128
# hidden_size = 512       #   LSTMのかくれ層次元  100 -> 128
# num_hidden  =  2        #   LSTM 1 層（東DL)   論文は２層
# qu_feature_size = 3072  #   questionの特徴量　画像と同じ

#try12
feature_size= 4096      #   VQAモデルへの特徴量
qu_vocab_size = 10000   #   単語サイズ　　10000
ans_vocab_size = 40897  #   分類するクラス　1000  40311  ->40897
word_embed = 256        #   単語の次元数　64 次元  -> 128  -> 64
hidden_size = 256       #   LSTMのかくれ層次元  100 -> 128
num_hidden  =  2        #   LSTM 1 層（東DL)   論文は２層
qu_feature_size = 4096  #   questionの特徴量　画像と同じ



#model = VQAModel_X(vocab_size=10000, n_answer=40311).to(device)  #n_answer は分類するクラス
model = VQAModel(feature_size=feature_size, qu_vocab_size=qu_vocab_size, ans_vocab_size=ans_vocab_size,
                 word_embed=word_embed, hidden_size=hidden_size, num_hidden=num_hidden).to(device)

# optimizer / criterion
num_epoch = 50
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

print("train start")
# train model
for epoch in range(num_epoch):
    train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
    print(f"【{epoch + 1}/{num_epoch}】\n"
          f"train time: {train_time:.2f} [s]\n"
          f"train loss: {train_loss:.4f}\n"
          f"train acc: {train_acc:.4f}\n"
          f"train simple acc: {train_simple_acc:.4f}")

####################################
# 提出用ファイルの作成
####################################
model.eval()
submission = []
for image, question , q_len in tqdm(test_loader):
    image, question = image.to(device), question.to(device)
    pred = model(image, question,q_len)
    pred = pred.argmax(1).cpu().item()
    submission.append(pred)

#submission = [train_dataset.idx2answer[id] for id in submission]
submission = [tokenize_to_words(token_dict.get_answer_index(id)) for id in submission]
submission = np.array(submission)
np.save("./テスト結果/submission302_Try12_45_last.npy", submission)
