######################################################################
#
#   DL基礎講座2024　最終課題「Visual Question Answering（VQA）」 400版
#
#   400シリーズは　各種ツールや実験機能の実装
#       追加　訓練データ作成を目的に作成
#       Bertを利用したImege処理　　質問・回答をLLMで生成する目的の事前確認
#       画像認識により、質問と回答を作れるかの確認
#   機能試験
#   1. train0000.jpg を読みこむ
#   2. 質問を入れる
#   3. 返事をもらう
#  注）PiPの途中で、colabからrestartの打診がくるがそのまま進める。ライブラリーの重複など
#######################################################################

##########################################
#       1 環境設定
#########################################
from google.colab import drive
drive.mount('/content/drive')
%cd '/content/drive/My Drive/Colab Notebooks/DL基礎講座2024/最終課題/VQA/'

!pip install langchain
!pip install --upgrade langchain


!pip install --upgrade langchain langchain-google-genai


!pip install google-generativeai
!pip install langchain_community

!pip install tqdm

!pip install --quiet langchain
!pip install --quiet langchain-google-genai
!pip install --quiet chromadb

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

from tqdm import tqdm
import time

#トークナイザーによる変更
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

#トークナイザー化で文字列のパディングを行う
from torch.nn.utils.rnn import pad_sequence

#gemini interface
# Gemini環境設定
import os
from google.colab import userdata
from google.colab import drive

import google.generativeai as genai

from langchain.chains import LLMChain

from langchain.prompts import PromptTemplate

from langchain.chat_models import ChatOpenAI

#from langchain.llms import Gemini


from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain.document_loaders import ImageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#########################################################################
#
#       1.画像判読の機能試験　　（train dataの回答作成　targetと比較）
#
#########################################################################

#Gemini環境設定

!pip install genai  # Make sure genai is installed
import genai
#from genai.chat import ChatModel  # Use the correct import path
#from genai.model import ChatModel

import os
import pandas
from PIL import Image
from langchain import PromptTemplate, LLMChain

!pip install google-generativeai
import google.generativeai as genai

import os
from google.colab import userdata

os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')
print(os.environ['GEMINI_API_KEY'] if os.environ['GEMINI_API_KEY'] else 'シークレットキーの追加ができていません．')

import google.generativeai as genai
import PIL.Image
import os
os.environ['GOOGLE_API_KEY'] = userdata.get('GEMINI_API_KEY')
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

prompt1 ="あなたは、視覚障害者のサポート者です。\
写真を分析し、視覚障がい者の質問答えてください。答えは簡潔に必要なところだけ話してください。\
写真は、上下逆さまとか、傾いていたり、欠けている事もあります。また写真の前後の関係は全くありません。\
その都度前の内容はリセットしてください。\
回答例として　yes/no  商品名　固有名称　雰囲気　数字　色などです。理由を考えて手短に一言で回答してください\
よく考えても答えがはっきりしないときは、「unanswerable」と答えてください\
ここ方はアメリカ人なので、英語で答えてください\
画像: {image} \
質問: {question}"

model = genai.GenerativeModel(model_name="gemini-1.5-flash")


#   trainのjsonをロードする。
df_path = "./data/train.json"
df_train = pandas.read_json(df_path)
df_train['answer'] = ""
submission = []

# 画像を読み込む
image_dir = "./data/train"
from google.generativeai.types import HarmCategory, HarmBlockThreshold

for idx  in range(20):     #先頭20件を処理する。

    image = PIL.Image.open(f"{image_dir}/{df_train['image'][idx]}")
    print(df_train['image'][idx])    #ファイル名

    # 質問リスト
    question = df_train['question'][idx]     #質問をセットする

    # 各質問に対して、LLM呼びだだし画像から推論

    if (idx != 9) or (idx != 10):
      answer = model.generate_content([prompt1, image, question])
      print('Q: {}'.format(question))         #質問を印刷
      print('A: {}'.format(answer.text))      #回答を印刷
      df_train['answer'][idx] = answer.text



#########################################################################
#
#       ２.訓練データの作成機能　　（train dataを使った新しいケース）
#
#########################################################################
#実行時のプロンプト

prompt1 ="あなたは、テストデータの作成者です。\
写真を分析し、この写真に関連した、質問と、回答を作ってください\
写真は、上下逆さまとか、傾いていたり、欠けている事もあります。また写真の前後の関係は全くありません。\
その都度前の内容はリセットしてください。\
回答として　yes/no  商品名　固有名称　雰囲気　数字　色などです。理由を考えて手短に一言で回答できる質問を作ってください。\
回答の種類は　yes/no に偏らないように 商品名　固有名称　雰囲気　数字　色など混ぜてバランスをとってください。\
画像は順番に２０個出します。\
回答者がわかりにくく、「unanswerable」と答える質問も含めてください。\
ここ方はアメリカ人に利用するので、英語で作成してください。\
画像: {image} \
質問: {question}"



model = genai.GenerativeModel(model_name="gemini-1.5-flash")


#   trainのjsonをロードする。
df_path = "./data/train.json"
df_train = pandas.read_json(df_path)
df_train['answer'] = ""
submission = []

# 画像を読み込む
image_dir = "./data/train"
from google.generativeai.types import HarmCategory, HarmBlockThreshold

for idx  in range(20):     #先頭20件を処理する。

    image = PIL.Image.open(f"{image_dir}/{df_train['image'][idx]}")
    print(df_train['image'][idx])    #ファイル名

    # 質問リスト
    question = "make question and answer"     #作業を指示する

    # 各質問に対して、LLM呼びだだし画像から推論

    if (idx != 9) or (idx != 10):
      answer = model.generate_content([prompt1, image, question])
      print('Q: {}'.format(question))         #質問を印刷
      print('A: {}'.format(answer.text))      #回答を印刷
      df_train['answer'][idx] = answer.text

