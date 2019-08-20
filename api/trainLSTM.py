from pyvi import ViTokenizer, ViPosTagger
import re 
from gensim.models import Word2Vec
import string
import codecs
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.models import load_model
import random

class Train:
  def __init__(self):
    self.maxlen_sentence = 100
    self.epochs = 10
    self.batch_size = 20
    self.num_class = 2
    self.dropout = 0.2
    self.modelw2v = None
    self.modelLSTM = None
    self.randText = None
    self.label = None


  def no_marks(self,s):
    VN_CHARS_LOWER = u'ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđð'
    VN_CHARS_UPPER = u'ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸÐĐ'
    VN_CHARS = VN_CHARS_LOWER + VN_CHARS_UPPER
    __INTAB = [ch for ch in VN_CHARS]
    __OUTTAB = "a"*17 + "o"*17 + "e"*11 + "u"*11 + "i"*5 + "y"*5 + "d"*2
    __OUTTAB += "A"*17 + "O"*17 + "E"*11 + "U"*11 + "I"*5 + "Y"*5 + "D"*2
    __r = re.compile("|".join(__INTAB))
    __replaces_dict = dict(zip(__INTAB, __OUTTAB))
    result = __r.sub(lambda m: __replaces_dict[m.group(0)], s)
    return result
  def normalize_text(self,text):
    path_nag = './data/nag1.txt'
    path_pos = './data/pos1.txt'
    path_not = './data/not.txt'
    with codecs.open(path_nag, 'r', encoding='UTF-8') as f:
        nag = f.readlines()
    nag_list1 = [' '+n.replace('\n', ' ') for n in nag]
    nag_list = [n.replace('\r','') for n in nag_list1]
    with codecs.open(path_pos, 'r', encoding='UTF-8') as f:
        pos = f.readlines()
    pos_list1 = [' '+n.replace('\n', ' ') for n in pos]
    pos_list = [n.replace('\r', '') for n in pos_list1]
    with codecs.open(path_not, 'r', encoding='UTF-8') as f:
        not_ = f.readlines()
    not_list1 = [ n.replace('\n', '') for n in not_]
    not_list = [n.replace('\r', '') for n in not_list1]
    #Remove các ký tự kéo dài: vd: đẹppppppp
    text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)
    # Chuyển thành chữ thường
    text = text.lower()

    #Chuẩn hóa tiếng Việt, xử lý emoj, chuẩn hóa tiếng Anh, thuật ngữ
    replace_list = {
        'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ', 'òe': 'oè', 'óe': 'oé','ỏe': 'oẻ',
        'õe': 'oẽ', 'ọe': 'oẹ', 'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ','ụy': 'uỵ','ủa':'uả',
        'ả': 'ả', 'ố': 'ố', 'u´': 'ố','ỗ': 'ỗ', 'ồ': 'ồ', 'ổ': 'ổ', 'ấ': 'ấ', 'ẫ': 'ẫ', 'ẩ': 'ẩ',
        'ầ': 'ầ', 'ỏ': 'ỏ', 'ề': 'ề','ễ': 'ễ', 'ắ': 'ắ', 'ủ': 'ủ', 'ế': 'ế', 'ở': 'ở', 'ỉ': 'ỉ',
        'ẻ': 'ẻ', 'àk': u' à ','aˋ': 'à', 'iˋ': 'ì', 'ă´': 'ắ','ử': 'ử', 'e˜': 'ẽ', 'y˜': 'ỹ', 'a´': 'á',
        #Quy các icon về 2 loại emoj: Tích cực hoặc tiêu cực
        "👹": "nagative", "👻": "positive", "💃": "positive",'🤙': ' positive ', '👍': ' positive ',
        "💄": "positive", "💎": "positive", "💩": "positive","😕": "nagative", "😱": "nagative", "😸": "positive",
        "😾": "nagative", "🚫": "nagative",  "🤬": "nagative","🧚": "positive", "🧡": "positive",'🐶':' positive ',
        '👎': ' nagative ', '😣': ' nagative ','✨': ' positive ', '❣': ' positive ','☀': ' positive ',
        '♥': ' positive ', '🤩': ' positive ', 'like': ' positive ', '💌': ' positive ',
        '🤣': ' positive ', '🖤': ' positive ', '🤤': ' positive ', ':(': ' nagative ', '😢': ' nagative ',
        '❤': ' positive ', '😍': ' positive ', '😘': ' positive ', '😪': ' nagative ', '😊': ' positive ',
        '?': ' ? ', '😁': ' positive ', '💖': ' positive ', '😟': ' nagative ', '😭': ' nagative ',
        '💯': ' positive ', '💗': ' positive ', '♡': ' positive ', '💜': ' positive ', '🤗': ' positive ',
        '^^': ' positive ', '😨': ' nagative ', '☺': ' positive ', '💋': ' positive ', '👌': ' positive ',
        '😖': ' nagative ', '😀': ' positive ', ':((': ' nagative ', '😡': ' nagative ', '😠': ' nagative ',
        '😒': ' nagative ', '🙂': ' positive ', '😏': ' nagative ', '😝': ' positive ', '😄': ' positive ',
        '😙': ' positive ', '😤': ' nagative ', '😎': ' positive ', '😆': ' positive ', '💚': ' positive ',
        '✌': ' positive ', '💕': ' positive ', '😞': ' nagative ', '😓': ' nagative ', '️🆗️': ' positive ',
        '😉': ' positive ', '😂': ' positive ', ':v': '  positive ', '=))': '  positive ', '😋': ' positive ',
        '💓': ' positive ', '😐': ' nagative ', ':3': ' positive ', '😫': ' nagative ', '😥': ' nagative ',
        '😃': ' positive ', '😬': ' 😬 ', '😌': ' 😌 ', '💛': ' positive ', '🤝': ' positive ', '🎈': ' positive ',
        '😗': ' positive ', '🤔': ' nagative ', '😑': ' nagative ', '🔥': ' nagative ', '🙏': ' nagative ',
        '🆗': ' positive ', '😻': ' positive ', '💙': ' positive ', '💟': ' positive ',
        '😚': ' positive ', '❌': ' nagative ', '👏': ' positive ', ';)': ' positive ', '<3': ' positive ',
        '🌝': ' positive ',  '🌷': ' positive ', '🌸': ' positive ', '🌺': ' positive ',
        '🌼': ' positive ', '🍓': ' positive ', '🐅': ' positive ', '🐾': ' positive ', '👉': ' positive ',
        '💐': ' positive ', '💞': ' positive ', '💥': ' positive ', '💪': ' positive ',
        '💰': ' positive ',  '😇': ' positive ', '😛': ' positive ', '😜': ' positive ',
        '🙃': ' positive ', '🤑': ' positive ', '🤪': ' positive ','☹': ' nagative ',  '💀': ' nagative ',
        '😔': ' nagative ', '😧': ' nagative ', '😩': ' nagative ', '😰': ' nagative ', '😳': ' nagative ',
        '😵': ' nagative ', '😶': ' nagative ', '🙁': ' nagative ', ':((' : ' nagative ', 
        #Chuẩn hóa 1 số sentiment words/English words
        ':))': '  positive ', ':)': ' positive ', 'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ',
        'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', ' oke ':  ' ok ',' okay':' ok ','okê':' ok ',
        ' tks ': u' cám ơn ', 'thks': u' cám ơn ', 'thanks': u' cám ơn ', 'ths': u' cám ơn ', 'thank': u' cám ơn ',
        '⭐': 'star ', '*': 'star ', '🌟': 'star ', '🎉': u' positive ',
        ' kg ': u' không ',' not ': u' không ', u' kg ': u' không ', '"k ': u' không ',' kh ':u' không ',' kô ':u' không ',' hok ':u' không ',' kp ': u' không phải ',u' kô ': u' không ', '"ko ': u' không ', u' ko ': u' không ', u' k ': u' không ', 'khong': u' không ', u' hok ': u' không ',
        'he he': ' positive ','hehe': ' positive ','hihi': ' positive ', 'haha': ' positive ', 'hjhj': ' positive ',
        ' lol ': ' nagative ',' cc ': ' nagative ',' cute ': u' dễ thương ',' huhu ': ' nagative ', ' vs ': u' với ', ' wa ': ' quá ', ' wá ': u' quá ', ' j ': u' gì ', '“': ' ',
        ' sz ': u' cỡ ', 'size': u' cỡ ', u' đx ': u' được ', 'dk ': u'được ', 'dc': u' được ', 'đk ': u'được ',
        ' đc ': u' được ',' authentic ': u' chuẩn chính hãng ',u' aut ': u' chuẩn chính hãng ', u' auth ': u' chuẩn chính hãng ', ' thick ': u' positive ', ' store ': u' cửa hàng ',
        ' shop ': u' cửa hàng ', ' sp ': u' sản phẩm ', ' gud ': u' tốt ',' god ': u' tốt ',' wel done ':' tốt ', ' good ': u' tốt ', ' gút ': u' tốt ',
        ' sấu' : u' xấu ',' gut' : u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', 'perfect': 'rất tốt', ' bt ': u' bình thường ',
        ' time ': u' thời gian ', ' qá' : u' quá ', u' ship ': u' giao hàng ', u' m ': u' mình ', u' mik ': u' mình ', u'quá hài lòg' : u'positive',
        'ể': 'ể', 'product': 'sản phẩm', 'quality': 'chất lượng','chat':' chất ', 'excelent': 'hoàn hảo', 'bad': 'tệ','fresh': ' tươi ','sad ': 'tệ ',
        ' date ': u' hạn sử dụng ', ' hsd ': u' hạn sử dụng ',' quickly ': u' nhanh ', ' quick ': u' nhanh ','fast': u' nhanh ',' delivery ': u' giao hàng ',u' síp ': u' giao hàng ',
        ' beautiful ': u' đẹp tuyệt vời ', u' tl ': u' trả lời ', u' r ': u' rồi ', u' shopE ': u' cửa hàng ',u' order ': u' đặt hàng ',
        ' chất lg ': u' chất lượng ',u' sd ': u' sử dụng ',u' dt ': u' điện thoại ',u' nt ': u' nhắn tin ',u' tl ': u' trả lời ',u' sài ': u' xài ',u'bjo':u' bao giờ ',
        ' thik ': u' thích ',u' sop ': u' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': u' rất ',u' quả ng ':u' quảng  ',
        ' dep ': u' đẹp ',u' xau ': u' xấu ',' delicious ': u' ngon ', u' hàg ': u' hàng ', u' qủa ': u' quả ',
        ' iu ': u' yêu ',' fake ': u' giả mạo ', 'trl': 'trả lời', '><': u' positive ', u' thích đáng ' : ' nagative ',
        ' por ': u' tệ ',' poor ': u' tệ ', ' ib ':u' nhắn tin ', ' rep ':u' trả lời ',u'fback':' feedback ',' fedback ':' feedback ',
        #dưới 3* quy về 1*, trên 3* quy về 5*
        '6 sao': ' 5star ','6 star': ' 5star ', '5star': ' 5star ','5 sao': ' 5star ','5sao': ' 5star ',
        'starstarstarstarstar': ' 5star ', '1 sao': ' 1star ', '1sao': ' 1star ','2 sao':' 1star ','2sao':' 1star ',
        '2 starstar':' 1star ','1star': ' 1star ', '0 sao': ' 1star ', '0star': ' 1star ',}

    for k, v in replace_list.items():
        text = text.replace(k, v)
        
  #    for k, v in replace_list.items():
  #        text = text.replace(k, v)
    
  #    for k in pos_list :
  #      text = text.replace(k, ' positive ')
  #    for k in nag_list :
  #      text = text.replace(k, ' nagative ')
    text = ViTokenizer.tokenize(text)
    
  #    for k, v in replace_list.items():
  #        text = text.replace(k, v)
        
    
    text = text.strip()
    texts = text.split()
    len_text = len(texts)
    for i in range(len_text):
        cp_text = texts[i]
        if cp_text in not_list: 
            numb_word = 2 if len_text - i - 1 >= 4 else len_text - i - 1
           
            for j in range(numb_word):
                if texts[i + j + 1] in pos_list or texts[i+j+1] == 'positive':
                    texts[i] = 'nagative'
                    texts[i + j + 1] = ''

                if texts[i + j + 1] in nag_list or texts[i+j+1] == 'nagative':
                    texts[i] = 'positive'
                    texts[i + j + 1] = ''
  #        else: 
  #            if cp_text in pos_list:
  # #                texts.append('positive')
  #                  texts[i] = 'positive'
  #            elif cp_text in nag_list:
  # #                texts.append('nagative')
  #                  texts[i] = 'nagative'

    text = u' '.join(texts)

    
    text = text.replace(u'"', u' ')
    text = text.replace(u'️', u'')
    text = text.replace('🏻','')
    # text = self.no_marks(text)
    return text
  #load data for word2vec train model 
  def load_data_word2vecTrain(self,path):
    X=[]
    result = []
    sentences= u""
    with open(path, encoding="utf8") as fr: 
      lines = fr.readlines()
      for line in lines: 
        line = re.sub('train_[0-9][0-9][0-9][0-9][0-9][0-9]', '', line)
        line = re.sub('\n','',line)
        sentences += line
        
    X1 = sentences.strip().split('"')
    index = 1 
    while index < len(X1) :
      X1[index] = X1[index].strip()
      if X1[index] != '0' and X1[index] != '1' :
        X1[index] = self.normalize_text(X1[index]) 
        X1[index] = re.sub('\{','',X1[index])
        X1[index] = re.sub('\}','',X1[index])
        X1[index] = re.sub('\=','',X1[index])
        X1[index] = re.sub('\(','',X1[index])
        X1[index] = re.sub('\)','',X1[index])
        X1[index] = re.sub('\,',' ',X1[index])
        if X1[index] != '' and X1[index] != ' ':
          X.append(X1[index])
          index +=1 
          while X1[index] != '0' and X1[index] != '1' :
            index +=1 
      index += 1
      
    for sentences in X: 
      for sentence in sentences.strip().split('.'):
        k = sentence.split()
        if k != []:
          result.append(k)
    return result


  #load data LSTM for train model 
  def load_data_LSTM(self,path, model_path):
    """
    input: 
    path: data path 
    model_path : word2vec model path 
    return : 
    S: 3D-array data convert to vector by word2vec model loaded in model_path
    L : 2D-array label of data 
    """
    X = [] 
    S = [] 
    L = [] 
    Z = []
    K = []
    sentences= ""
    with open(path,encoding="utf8") as fr: 
      lines = fr.readlines()
      for line in lines: 
        line = re.sub('\n','',line)
        line = line.strip()
        sentences += line
    X1 = re.compile('train_[0-9][0-9][0-9][0-9][0-9][0-9]').split(sentences) 
    index = 1 
    while index < len(X1) :
      i = 0
      while X1[index][len(X1[index]) - 1-i] != '1' and X1[index][len(X1[index]) - 1-i]!= '0' :
        i +=1 
      tmp = [0,0]
      tmp[int(X1[index][len(X1[index]) - 1-i],10)] = 1
      L.append(tmp)
      
      X1[index] = X1[index][:len(X1[index]) - 1-i]
      K.append(X1[index])
      X1[index] = re.sub('\.',' ',X1[index])
      X1[index] = self.normalize_text(X1[index]) 
      X1[index] = re.sub('\{','',X1[index])
      X1[index] = re.sub('\}','',X1[index])
      X1[index] = re.sub('\=','',X1[index])
      X1[index] = re.sub('\(','',X1[index])
      X1[index] = re.sub('\)','',X1[index])
      X1[index] = re.sub('\,',' ',X1[index])
      
        
      X.append(X1[index])
      index += 1
  
    if self.modelw2v == None:
      self.modelw2v = Word2Vec.load(model_path)
    model = self.modelw2v
    for line in X :
      tmp = []
      for w in line.split():
        if w != ' ' :
          if w in model.wv :
            tmp.append(model.wv[w])
          else :
            tmp.append(np.zeros(shape=(model.wv[model.wv.index2word[0]].shape[0],), dtype=float))
      Z.append(line)
      if len(tmp) > self.maxlen_sentence :
        tmp = tmp[:self.maxlen_sentence]
      elif len(tmp) < self.maxlen_sentence :
        if len(tmp) == 0:
          tmp= np.zeros(shape=(1,100), dtype=float)
        tmp = np.concatenate((tmp, np.zeros(shape=(self.maxlen_sentence - len(tmp),100), dtype=float)),axis=0)
      S.append(tmp)
       
    return np.array(S),np.array(L), Z, K
  #load test data 
  def load_data_test(self,path_data, model_path):
    X = [] 
    S = [] 
    
    sentences= ""
    with open(path_data) as fr: 
      lines = fr.readlines()
      for line in lines: 
        line = re.sub('\n','',line)
        sentences += line
        
    X1 = re.compile('test_[0-9][0-9][0-9][0-9][0-9][0-9]').split(sentences)    
    index = 1 
    while index < len(X1) :
      X1[index] = re.sub('\"','',X1[index])
      
      X1[index] = re.sub('\.',' ',X1[index])
      X1[index] = self.normalize_text(X1[index]) 
      X1[index] = re.sub('\{','',X1[index])
      X1[index] = re.sub('\}','',X1[index])
      X1[index] = re.sub('\=','',X1[index])
      X1[index] = re.sub('\(','',X1[index])
      X1[index] = re.sub('\)','',X1[index])
      X1[index] = re.sub('\,',' ',X1[index])
        
      X.append(X1[index])
      index += 1
    
    if self.modelw2v == None:
      self.modelw2v = Word2Vec.load(model_path)
    model = self.modelw2v
    for line in X :
      tmp = []
      for w in line.split():
        if w != ' ' :
          if w in model.wv :
            tmp.append(model.wv[w])
          else :
            tmp.append(np.zeros(shape=(model.wv[model.wv.index2word[0]].shape[0],), dtype=float))
      if len(tmp) > 40:
        tmp = tmp[:40]
      elif len(tmp) < 40:
        if len(tmp) == 0:
          tmp= np.zeros(shape=(1,100), dtype=float)
        tmp = np.concatenate((tmp, np.zeros(shape=(40 - len(tmp),100), dtype=float)),axis=0)
      S.append(tmp)
       
    return np.array(S)
  #train model LSTM 
  def train_model_LSTM(self,path_model,X,Y):
    """
    path_model : path for save model 
    X : 3D-array data content 
    Y : 2D-array label 
    """
    model = Sequential()
    inputdim = (X.shape[1], X.shape[2])
    model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(self.dropout))
    model.add(LSTM(32))
    model.add(Dense(self.num_class, activation="softmax"))
  
    model.compile(loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy'])
    # model.load_weights('./data/LSTM.model73')
    
    model.fit(X, Y, batch_size=self.batch_size, epochs=self.epochs)
    model.save_weights(path_model)

  #predict data 
  def predict(self,path_data_test,model_path_word2vec, model_path_LSTM):
    X,Y,Z,K = self.load_data_LSTM(path_data_test, model_path_word2vec)
    X = X[int(len(X)*0.8):]
    Y = Y[int(len(Y)*0.8):]
    Z = Z[int(len(Z)*0.8):]
    K = K[int(len(K)*0.8):]
    inputdim = (X.shape[1], X.shape[2])
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=inputdim))
    model.add(Dropout(self.dropout))
    model.add(LSTM(32))
    model.add(Dense(self.num_class, activation="softmax"))
  
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.load_weights(model_path_LSTM)
    
    #  X = load_data_test(path_data_test,model_path_word2vec)
    
    yhat = model.predict(np.array(X))
    yhat = np.argmax(yhat, axis=1)
    num = 0 
    submit = []
    
    #  with open('/content/drive/My Drive/colab/sample_submission.csv') as fr:
    #    lines = fr.readlines()
    #    for line in lines:
    #      line = re.sub('test_[0-9][0-9][0-9][0-9][0-9][0-9]','',line)
    #      line = re.sub('\,','',line)
    #      if(int(line) == 0 or int(line) == 1) :
    #        submit.append(int(line))
    #  for i in range(len(yhat)):
    #    if yhat[i] == submit[i]  :
    #      num += 1
  
    for i in range(len(yhat)):
      if Y[i][yhat[i]] == 1 :
        num += 1 
      else :
        print(i)
        print(Z[i])
        print(K[i])
        print(yhat[i])
        print(Y[i])
        
    print(len(yhat))
    print(len(Y))
    print(len(Z))
    print(len(X))
    print("predict : " + str(num))
    print(num/len(yhat))
    print(yhat[:20])
  #predict sentence 
  def convert_sentence_tovec(self,text, model_path_word2vec):
    
    text = re.sub('\"','',text)
    text = re.sub('\.',' ',text)
    text = self.normalize_text(text) 
    text = re.sub('\{','',text)
    text = re.sub('\}','',text)
    text = re.sub('\=','',text)
    text = re.sub('\(','',text)
    text = re.sub('\)','',text)
    text = re.sub('\,',' ',text)

    if self.modelw2v == None:
      self.modelw2v = Word2Vec.load(model_path_word2vec)
    modelw2v = self.modelw2v
    Wvec = []
    tmp = []
    for w in text.split():
      if w != ' ' :
        if w in modelw2v.wv :
          tmp.append(modelw2v.wv[w])
        else :
          tmp.append(np.zeros(shape=(modelw2v.wv[modelw2v.wv.index2word[0]].shape[0],), dtype=float))
    if len(tmp) > 40:
      tmp = tmp[:40]
    elif len(tmp) < 40:
      if len(tmp) == 0:
        tmp= np.zeros(shape=(1,100), dtype=float)
      tmp = np.concatenate((tmp, np.zeros(shape=(40 - len(tmp),100), dtype=float)),axis=0)
    Wvec.append(tmp)
    return np.array(Wvec)


  #predic sentence converted to vector by word2vec and return 
  def predict_sentence(self, text,model_path_word2vec, model_path_LSTM):
    Wvec = self.convert_sentence_tovec(text,model_path_word2vec)
    if self.modelLSTM == None:

      inputdim = (Wvec.shape[1], Wvec.shape[2])
      self.modelLSTM = Sequential()
      self.modelLSTM.add(LSTM(64, return_sequences=True, input_shape=inputdim))
      self.modelLSTM.add(Dropout(self.dropout))
      self.modelLSTM.add(LSTM(32))
      self.modelLSTM.add(Dense(self.num_class, activation="softmax"))
      self.modelLSTM.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.Adadelta(),
                            metrics=['accuracy'])
      self.modelLSTM.load_weights(model_path_LSTM)

    model = self.modelLSTM
    yhat = model.predict(np.array(Wvec))
    yhat = np.argmax(yhat, axis=1)
    return yhat[0] 



  #get randomtext 
  def get_RanText(self,pathdata):
    if self.randText == None:
      sentences= ""
      X  =[]
      L = []
      with open(pathdata,encoding="utf8") as fr: 
        lines = fr.readlines()
        for line in lines: 
          line = re.sub('\n','',line)
          line = line.strip()
          sentences += line
      X1 = re.compile('train_[0-9][0-9][0-9][0-9][0-9][0-9]').split(sentences)
      X1 = X1[int(0.8*len(X1)) :] 
      index = 1 
      while index < len(X1) :
        i = 0
        while X1[index][len(X1[index]) - 1-i] != '1' and X1[index][len(X1[index]) - 1-i]!= '0' :
          i +=1 
        tmp = [0,0]
        tmp[int(X1[index][len(X1[index]) - 1-i],10)] = 1
        L.append(tmp)
        X1[index] = X1[index][:len(X1[index]) - 1-i]
        
        X.append(X1[index])
        index += 1
      self.randText = X
      self.label = L
    index = random.randint(0,len(self.randText))
    return self.randText[index], self.label[index]

if __name__ == "__main__":
  mod = Train()
  text = mod.load_data_word2vecTrain('./data/train.crash')
  model = Word2Vec(text, size=100, window=5, min_count=1, workers=4)
  model.save('./data/word2vec.model2')
  X,Y ,Z,K= mod.load_data_LSTM('./data/train.crash','./data/word2vec.model2')
  X = X[:int(len(X)*0.8)]
  Y = Y[:int(len(Y)*0.8)]
  mod.train_model_LSTM('./data/LSTM.model73',X,Y)
  mod.predict('./data/train.crash','./data/word2vec.model2','./data/LSTM.model73')
  # if mod.predict_sentence("shop phuc vu kem, san pham khong duoc nhu mong doi :( ",'./data/word2vec.model2','./data/LSTM.model73') == 1:
  #   print("tieu cuc")
  # else:
  #   print("tic cuc")
  pass