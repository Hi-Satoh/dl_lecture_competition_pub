#######################################################################
#
#   DL基礎講座2024　最終課題「Visual Question Answering（VQA）」 VQA303
#
#   自然言語処理　　BertTokenizer ＋ Bert_Model辞書
#   モデル        seq2seq + Image  VQAシステム（Encoder + Decoder)
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

#######################################
#    1-3 ライブラリーの設定
#######################################
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

!pip install torch transformers

import torch
from transformers import BertTokenizer, BertModel

#######################################################################
#
#  2. 自然言語処理 プラットホーム　（Bertのオークナイザーを全面的に利用）
#   0.訓練済みモデルの利用
#   1.必要機能の共通関数化
#   2.その他サービス機能
#
#######################################################################

# Bert Tokenizer-kenizer環境の設定
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

pad_token_id = tokenizer.pad_token_id                                         #padding Token文字
question_max_length = 65                                                      #paddingの長さ65
anser_max_length = 10                                                         #paddingの長さ65


# テキストをトークナイズし、トークンサイズを取得
def tokenize_texts(texts, tokenizer):
    tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
    token_lengths = [len(tokens) for tokens in tokenized_texts]
    return tokenized_texts, token_lengths

def tokenize_texts_2(texts, tokenizer):
    tokenized_texts = [tokenizer.encode(text, skip_special_tokens=True) for text in texts]
    token_lengths = [len(tokens) for tokens in tokenized_texts]
    return tokenized_texts, token_lengths


# 最大シーケンス長に基づいてパディング (短くすることを考慮）)
def pad_texts(tokenized_texts, max_length, pad_token_id):
    padded_texts = [tokens[:max_length] + [pad_token_id] * max(0, max_length - len(tokens)) for tokens in tokenized_texts]
    return padded_texts

# テキストを埋め込みベクトルに変換
def convert_text_to_embeddings(padded_texts, tokenizer, model):
    input_ids = torch.tensor(padded_texts)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state


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



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#######################################
#  　Debug時の為 旧システム参照用として残す。
#######################################
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


#################################
#    3-1 データローダーのクラス
#     Base Modelの機能を継承
#################################
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir,  tokenizer, transform=None, answer=True):
        self.transform = transform            # 画像の前処理
        self.image_dir = image_dir            # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)   # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer
        self.tokenizer = tokenizer                   ########トークナイザーにする。
        #self.token_dict = token_dict                ## 独自　辞書の削除

        self.max_question_length = get_max_seq_length(self.df, "question", tokenizer)  ######　questionの最大トークン数
        print("max_question_length =", self.max_question_length)
        self.max_answer_length  = 10                             #get_max_seq_length(self.df, "answers" , tokenizer)  ######　answerの最大トークン数
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
        question : -> vocab_sizeはトークンに長さになる
            質問文”トークン化"
        answers :
            トークンインデックスになる
        mode_answer_idx : torch.Tensor  (1)
            トークンインデックスになる
        """
        #イメージ読み込み
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)

        #質問文をトークナイズする
        question = process_text(self.df["question"][idx])                       #当課題で前提となる基本処理を行う。（トークナイザーと被る可能性はある）

        #回答文
        if self.answer:
            answers = [process_text(answer["answer"]) for answer in self.df["answers"][idx]]
            #print("answers",answers)
            mode_answer = mode(answers) # 最頻値を取得（正解ラベル）
            #print("mode_answer",mode_answer)
            return image, question, answers, mode_answer
        else:

            return image, question

    def max_question_token_count(self):
      return self.max_question_length

    def max_answer_count(self):
      return self.max_answer_length

    def __len__(self):
        return len(self.df)


#########################################
#   3-2 datasetの準備
#   dataloader / model
#   データ件数　Train 19,873   test 4,969
#########################################
batch_size = 30       #128 -> 64

# VGG19用の前処理を定義
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # ピクセル値をテンソルに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
])

Path = "/content/data/"     #ランタイム化
#Path = "/content/drive/My Drive/Colab Notebooks/DL基礎講座2024/最終課題/VQA/data/"

train_path = Path + "train.json"
valid_path = Path + "valid.json"
image_dir_train_path = Path + "train"
image_dir_valid_path = Path + "valid"

train_dataset = VQADataset(df_path="./data/train.json", image_dir=image_dir_train_path, tokenizer=tokenizer, transform=transform)
test_dataset = VQADataset(df_path="./data/valid.json", image_dir=image_dir_valid_path,  tokenizer=tokenizer, transform=transform, answer=False)
#test_dataset.update_dict(train_dataset)

#Full data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  #!!!!!!!
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


### test関連で追加した内容
#ロジックテスト用の少ないデータセット
#多すぎるので、少しにする　（動作確認用）
#N = 300  # サンプリングするデータ数
# subset_train_indices = range(0,5000)
# subset_test_indices = range(0,2000)

# subset_train = torch.utils.data.Subset(train_dataset, subset_train_indices)
# subset_test = torch.utils.data.Subset(test_dataset, subset_test_indices)

# train_loader = torch.utils.data.DataLoader(subset_train, batch_size=batch_size, shuffle=False)   #!!!!!!!
# test_loader = torch.utils.data.DataLoader(subset_test, batch_size=1, shuffle=False)
#下記は一時的
#rev_test_dataset = VQADataset(df_path="./data/train.json", image_dir=image_dir_train_path, tokenizer=tokenizer, transform=transform, answer=False)
#rev_test_loader = torch.utils.data.DataLoader(rev_test_dataset, batch_size=1, shuffle=False)


#############################
# 4. VQAクラス・関数          #
#                           #
#############################
#########################
# 4-1. 評価指標の実装     #
#                       #
#########################

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


########################################
#   パラメータ
#     word_num = 単語辞書の大きさ             voca_size (10000)
#     emb_dim  = 単語列に長さ（入力トークン数） question最大長さ　(57) XXXXXX間違え
#     hid_dim  = LSTMのh                    調整値　現在　　(100)
#     text_feat = VQAに渡す特徴量ベクトル　　　　調整値　現在　 (512)
########################################

class textEncoder_x(nn.Module):
    #def __init__(self, word_num, emb_dim, hid_dim, text_feat):
    def __init__(self, qu_vocab_size, word_embed, hidden_size, num_hidden, qu_feature_size):
        super().__init__()
        print("textEncorder_x")
        #self.emb = nn.Embedding(word_num, emb_dim)
        #self.emb = nn.Embedding(10000, word_embed)       #削除 !!!!!!
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed, hidden_size, num_hidden, batch_first=True)  # nn.LSTMの使用  注意!!!!!!!　batch first
        #self.lstm = nn.LSTM(64, hid_dim, 1, batch_first=True)  # nn.LSTMの使用
        self.linear = nn.Linear(2*hidden_size, qu_feature_size)

    def forward(self, x,  len_seq, len_seq_max=0, init_state=None):
        #print("encoder"
        #print(x)
        #h = self.emb(x)       ＃削除 !!!!           #inp (batch_size=128, msg_len=65) => (batch_size_128,msg_len=65 word_embed=128)　　　　　(128, 65, 128)
        h = self.tanh(x)                  ##  -1 から　+1  平準化
        #print("encoded after tanh")
        #print(h)
        if len_seq_max > 0:
            #h, _ = self.lstm(h[:, 0:len_seq_max, :], init_state)          #h (batch_n ,emb, num_hodden )  h = (128, 65, 100)   cell = (層, batch_n, hidden_n) (1,128. 100)
            h, (hx, cell) = self.lstm(h[:, 0:len_seq_max, :], init_state)
        else:
            #h, _ = self.lstm(h, init_state)
            h, (hx, cell) = self.lstm(h, init_state)                      #h (batch_n ,emb, num_hodden )  h = (128, 65, 100)   cell = (1, 128, 100)
        h = h.transpose(0, 1)                                             #h (batch_n ,emb, num_hodden )  h = (65, 128, 100)   cell = (1, 128, 100)

        if len_seq is not None:
            #print("len_seq", len_seq)
            #print("len(x)",len(x))
            #print("h",h)
            h = h[len_seq - 1, list(range(len(x))), :]                    #(128,word_emb, num_hodden )  h = (1, 128, 100)   cell = (1, 128, 1,100)
        else:
            h = h[-1]                                                     #(128,word_emb, num_hodden )  h = (128, 100)   cell = (1, 128, 1,100)
        cell = cell[-1]                                                   #cell = (128, 1,100)
        h = torch.cat((h, cell), dim=1)                #(128,word_emb, num_hodden )  h = (128, 100) +  cell = (1, 128, 1,100)
        # h = h.transpose(0, 1)                     # (128, 1, 100)     batch_sizeを戻す
        h = h.reshape(h.size()[0], -1)   # (128, 2*num_layer*hidden_size=200)　　　　　2layerの次元削減

        y = self.linear(h)
        #print("encoder exit", y )
        return y

###########################################
#      画像エンコーダー　VGG19          　　　#
#      トレーニング済みクラス                 #
###########################################

class ImgEncoder(nn.Module):

    def __init__(self, embed_dim):   #=image_feature

        super(ImgEncoder, self).__init__()
        self.model = models.vgg19(pretrained=True)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1]) # remove vgg19 last layer
        self.fc = nn.Linear(in_features, embed_dim)     #embeded_dimはbertのdimに合わせる。　embed_dim=768
        self.activation = nn.Tanh()  # 活性化関数
        print("vgg19 encorder")

    def forward(self, image):

        with torch.no_grad():
            img_feature = self.model(image) # (batch, channel, height, width)
        img_feature = self.fc(img_feature)

        l2_norm = F.normalize(img_feature, p=2, dim=1).detach()        #これでもいいかも
        #l2_norm = self.activation(img_feature)                          #おそらくtanhが必要
        return l2_norm


###########################################
#      テキストエンコーダー　クラス・関数　　　　#
###########################################

class textEncoder(nn.Module):
    def __init__(self, bert_model, hidden_dim):
        super(textEncoder, self).__init__()
        #self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert = bert_model
        self.transform = nn.Linear(self.bert.config.hidden_size, hidden_dim)         #hidden_dimは小さくていい
        #print('textE1')
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        #print('textE2')
        transformed_output = self.transform(pooled_output)
        #print('textE3')
        return transformed_output

###########################################
#      テキストデコーダー　クラス・関数 　　　　#
###########################################

class textDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(textDecoder, self).__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        #print('textDe1')
    def forward(self, x, hidden):
        #print('textDe2')
        hidden = hidden.to(x.device)
        outputs, hidden = self.rnn(x, hidden)           #output (batch_size, seq_len, hidden_size)  hidden (num_layer, batch_size,hidden_size　vector)
        outputs = self.fc(outputs)
        #print("exit textD3")
        return outputs, hidden


##########################################
# 　　Attention機能                       #
##########################################

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

def create_attention_mask(question_mask, img_token_count=1):     #image Token化してattention maskを作る
    batch_size, seq_len = question_mask.size()
    # Create a mask for the image tokens
    img_mask = torch.ones(batch_size, img_token_count, dtype=question_mask.dtype, device=question_mask.device)
    # Concatenate the image mask and the question mask
    combined_mask = torch.cat((img_mask, question_mask), dim=1)  # (batch_size, seq_len + img_token_count)
    combined_mask = combined_mask.unsqueeze(1)  # (batch_size, 1, seq_len + img_token_count)
    return combined_mask


##########################################
# 　　VQAクラス                            #
##########################################

class VQAModel(nn.Module):
    def __init__(self, feature_size, qu_vocab_size, ans_vocab_size, word_embed, hidden_size, num_hidden,output_dim, bert_model):
        super(VQAModel, self).__init__()

        self.img_encoder = ImgEncoder(hidden_size)     #現在はVGG19
        self.qu_encoder = textEncoder(bert_model, hidden_size)
        self.an_decoder = textDecoder(hidden_size,output_dim)                   #output_dimはbertの辞書サイズ(=30522) 大きすぎるので変更

        #self.attention = MultiHeadAttention(hidden_size, heads=4)    #self attentionに変更
        self.attention = SelfAttention(hidden_size)  # Self-Attention instead of Multi-Head Attention

        # self.dropout = nn.Dropout(0.5)
        # self.tanh = nn.Tanh()
        # self.fc1 = nn.Linear(feature_size, feature_size)
        # self.fc2 = nn.Linear(feature_size, ans_vocab_size)

    def forward(self, image, question, q_len):
        #####
        #   image   　画像データ(244,244)
        #   question  分散表現のquestion
        #   q_len     batch単位でpaddingされているので、seqの長さ情報（maskと同じ働き）
        #####
        img_feature = self.img_encoder(image)               # (batchsize, 1, feature_size=768)   1 Token分の情報
        text_outputs = self.qu_encoder(question,q_len)      # (batchsize, 1, seq_l,  hidden_size)
        # 特徴量の混合(画像特徴をシーケンスに埋め込み)　1 token分の情報にする

        #combined_feature = img_feature * qst_feature
        img_feature = img_feature.unsqueeze(1)       # (batch_size, 1, hidden_dim=768 )
        combined_features = torch.cat((img_feature, text_outputs), dim=1)  # (batch_size, seq_len + 1, hidden_dim)
        # mask (q_len)を画像のToken分、加算する

        #（追加）
        # # Attention Mechanism
        # attention_mask = torch.ones((combined_features.size(0), combined_features.size(1)), dtype=torch.bool).to(combined_features.device)
        # combined_features = self.attention(combined_features, combined_features, combined_features, attention_mask)

        batch_size_w = len(question)
        img_mask = torch.ones((batch_size_w, 1), dtype=q_len.dtype).to(image.device)
        adjusted_attention_mask = torch.cat((img_mask, q_len), dim=1)  # (batch_size, total_seq_length + 1)　#1toke加算したmask

        #Create attention mask
        combined_mask = create_attention_mask(q_len, img_token_count=1)

        # Apply self-attention
        combined_features = self.attention(combined_features, combined_mask)  # Apply self-attention

        # デコーダーによる回答生成
        h0 = torch.zeros(num_hidden, combined_features.size(0), hidden_size).to(combined_features.device)    #num_layersはnum_hiddenと同じ値
        output, _ = self.an_decoder(combined_features, h0)           #戻り値　GRUを通している　output (batch_size, seq_len, output_dim)  output_dim=辞書の大きさ

        return output, adjusted_attention_mask

#######################################
#   5.訓練関連の関数                     #
#      訓練時に必要な言語処理関数も含まれる　#
#                                     #
#######################################
# 4. 学習の実装

import torch.nn.functional as F
def rearrange_data(data):   #dataloaderの改修せずに吸収（answersの取り出し並べ替え）もう少しいい書き方がいい
    # 各カテゴリのリストを初期化
    category1 = []
    category2 = []
    category3 = []
    category4 = []
    category5 = []
    category6 = []
    category7 = []
    category8 = []
    category9 = []
    category10 = []

    # 各データポイントを順番に抽出
    for item in data:
        category1.append(item[0])
        category2.append(item[1])
        category3.append(item[2])
        category4.append(item[3])
        category5.append(item[4])
        category6.append(item[5])
        category7.append(item[6])
        category8.append(item[7])
        category9.append(item[8])
        category10.append(item[9])

    # 新しいリストに並べ替えたデータを格納
    rearranged_data = [tuple(category1), tuple(category2), tuple(category3),
                       tuple(category4), tuple(category5), tuple(category6),
                       tuple(category7), tuple(category8), tuple(category9),
                       tuple(category10)
                       ]

    return rearranged_data



#outputにある単語インデックスからトークンのインデックス→文字列の生成
def generate_predictions(outputs, tokenizer, attention_mask):
    # Softmaxを適用して確率に変換
    probabilities = F.softmax(outputs, dim=-1)

    # 最も確率の高いトークンのインデックスを取得
    _, predicted_ids = torch.max(probabilities, dim=-1)

    # パディングトークンのインデックスを無視
    predicted_tokens = []
    for i, ids in enumerate(predicted_ids):
        tokens = []
        for j, token_id in enumerate(ids):
            if attention_mask[i, j] != 0:  # パディングでないトークンのみを対象とする
                token = tokenizer.convert_ids_to_tokens([token_id])[0]
                if token not in ["[CLS]", "[SEP]", "[PAD]"]:  # 特殊トークンを除外
                    tokens.append(token)

        # サブワードトークンを再結合してテキストに変換
        text = reconstruct_text_from_tokens(tokens)
        # if text == "" or text == "una" or text == "unans" or text == "unanswer" :
        #     text = "unanswerable"   #意図的に変更しないことにする。
        predicted_tokens.append(text)

    return predicted_tokens

def reconstruct_text_from_tokens(tokens):
    text = ""
    for token in tokens:
        if token.startswith("##"):
            text += token[2:]
        else:
            text += " " + token
    return text.strip()


def tokenize_targets(target_list, tokenizer, max_length=20):      #トークナイスする関数
    tokenized_targets = []
    for target in target_list:
        encoded_target = tokenizer(target, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        tokenized_targets.append(encoded_target['input_ids'].squeeze(0))
    return torch.stack(tokenized_targets)


def calculate_accuracy(a, b):     #サービス関数  タプルの比較関数　mode処理
    # リストとタプルの長さが異なる場合、短い方に合わせる
    min_length = min(len(a), len(b))
    # 比較する範囲を短い方の長さに制限
    a = a[:min_length]
    b = b[:min_length]
    # 正解数をカウント
    correct_count = sum(1 for x, y in zip(a, b) if x == y)
    # 正解率を計算
    accuracy = correct_count / min_length
    return accuracy


######################################
#                                    #
#    訓練の実施関数                    #
#                                    #
######################################

def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    cnt = 0
    for image, question, answers, mode_answer in tqdm(dataloader):
        question_list = list(question)
        # Bert Tokenizerを通す
        question_tokens = tokenizer(question_list, padding='max_length', truncation=True, max_length=question_max_length, return_tensors='pt')
        question = question_tokens['input_ids']
        q_len = question_tokens['attention_mask']

        image, question, q_len = \
            image.to(device), question.to(device), q_len.to(device)

        pred, adjusted_attention_mask = model(image, question, q_len)                   #question　　（batch _size, seq_n(固定65?VQADataの指定), dim(固定 768)

        # 分散表現としての出力を取得
        outputs = pred   #.squeeze(1)  # (batch_size, seq_length, output_dim) output_dimは辞書サイズでいいのか？

        ####### 損失の計算のためのtraget ##########重要
        target_list = list(mode_answer)
        target_list = tokenize_targets(target_list, tokenizer, question_max_length+1).to(device)

        # outputs の形状を (batch_size * total_seq_length, output_dim) に変形
        batch_size, total_seq_length, output_dim = outputs.size()
        outputs = outputs.view(batch_size * total_seq_length, output_dim)

        # answers の形状を (batch_size * total_seq_length,) に変形
        target_list = target_list.view(batch_size * total_seq_length)

        #lossを計算
        loss = criterion(outputs, target_list)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        #統計処理
        total_loss += loss.item()
        # 最も高い確率のトークンIDを取得
        outputs = outputs.view(batch_size, total_seq_length, output_dim)  # 元の形状に戻す
        predictions = generate_predictions(outputs, tokenizer, adjusted_attention_mask)

        answers = rearrange_data(answers)      #データ配列補正
        total_acc += VQA_criterion(predictions, answers)  # VQA accuracyの ここはこける
        simple_acc +=  calculate_accuracy(predictions, mode_answer)  # simple accuracy


    #     # Debug用　内部進捗確認用のプリント機能
    #     if (cnt % 300 == 0) & (cnt > 299) :
    #         print("predictions",predictions)
    #         print("mode_answers",mode_answer)
    #         print("simple_acc",simple_acc)

    #         print("answers",answers)


    #     cnt += 1
    # print("predictions",predictions)
    # print("answers",answers)
    # print("mode_answers",mode_answer)

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start



def eval(model, dataloader, optimizer, criterion, device):       #基本的に利用していない 必要に応じて改修
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, q_len, answers, mode_answer in tqdm(dataloader):
        image, question, q_len, answer, mode_answer = \
            image.to(device), question.to(device),  q_len.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question, q_len)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


######################################
######################################
#                                    #
#     ６。訓練の実施 コード             #
#                                    #
######################################                                   #
######################################

from tqdm import tqdm_notebook as tqdm
set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# before Try5
# feature_size= 1024      #   VQAモデルへの特徴量
# qu_vocab_size = 10000   #   単語サイズ　　10000
# ans_vocab_size = 30522  #   分類するクラス　1000  40311  ->40897 -> bert 辞書サイズ　30522
# word_embed = 768        #   単語の次元数　64 次元  -> 128  -> 64  ->bert tokenizer 768
# hidden_size = 768       #   LSTMのかくれ層次元  100 -> 128 ->768  encode ,decodeの次元
# num_hidden  =  1        #    LSTM 1 層（東DL)   論文は２層
# qu_feature_size = 1024  #   questionの特徴量　画像と同じ
# output_dim = 30522      #   Tokenizerの辞書の長さ -> bert 辞書サイズ　30522    重複を避けたい　整理!

#Try 6
feature_size= 4096      #   VQAモデルへの特徴量
qu_vocab_size = 10000   #   単語サイズ　　10000
ans_vocab_size = 30522  #   分類するクラス　1000  40311  ->40897 -> bert 辞書サイズ　30522
word_embed = 768        #   単語の次元数　64 次元  -> 128  -> 64  ->bert tokenizer 768
hidden_size = 768       #   LSTMのかくれ層次元  100 -> 128 ->768  encode ,decodeの次元
num_hidden  =  1        #    LSTM 1 層（東DL)   論文は２層
qu_feature_size = 4096  #   questionの特徴量　画像と同じ
output_dim = 30522      #   Tokenizerの辞書の長さ -> bert 辞書サイズ　30522    重複を避けたい　整理!


model = VQAModel(feature_size=feature_size, qu_vocab_size=qu_vocab_size, ans_vocab_size=ans_vocab_size,
                 word_embed=word_embed, hidden_size=hidden_size, num_hidden=num_hidden, output_dim=output_dim, bert_model=bert_model).to(device)

# optimizer / criterion
num_epoch = 40   #Epoch 実験に応じ変更
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
# 訓練後のモデル保存
####################################

torch.save(model.state_dict(), "./model/model303_Try6_40_final.pth")


####################################
#
#       提出用ファイルの作成
#
####################################

model.eval()
submission = []
for image, question  in tqdm(test_loader):
        question_list = list(question)
        question_tokens = tokenizer(question_list, padding='max_length', truncation=True, max_length=question_max_length, return_tensors='pt')
        question = question_tokens['input_ids']
        q_len = question_tokens['attention_mask']
        image, question, q_len = \
            image.to(device), question.to(device), q_len.to(device)

        pred, adjusted_attention_mask = model(image, question, q_len)                   #question　　（batch _size, seq_n(固定65?VQADataの指定), dim(固定 768)
        outputs = pred   #.squeeze(1)  # (batch_size, seq_length, output_dim) output_dimは辞書サイズでいいのか？

        # 最も高い確率のトークンIDを取得
        predicted_texts = generate_predictions(outputs, tokenizer, adjusted_attention_mask)
        submission.append(predicted_texts)
submission = np.array(submission)
np.save("./テスト結果/submission303_Try9_40.npy", submission)
