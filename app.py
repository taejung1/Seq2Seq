import sys
#print(sys.version)

# 챗봇 라이브러리 불러오기


from Korpora import KoreanChatbotKorpus
corpus = KoreanChatbotKorpus()

# python app.py

# print(corpus.get_all_texts()[:5])
# print(corpus.get_all_pairs()[:5])

len(corpus.get_all_texts())

texts = []
pairs = []
for i, (text, pair) in enumerate(zip(corpus.get_all_texts(), corpus.get_all_pairs())):
    texts.append(text)
    pairs.append(pair)
    if i >= 2000: 
        break 

list(zip(texts, pairs))[1995:2000]

import re
def clean_sentence(sentence):
    # 한글, 숫자를 제외한 모든 문자는 제거합니다.
    sentence = re.sub(r'[^0-9ㄱ-ㅎㅏ-ㅣ가-힣 ]',r'', sentence)
    return sentence

# print(clean_sentence('안녕하세요~:)'))
# print(clean_sentence('텐서플로^@^%#@!'))

from konlpy.tag import Okt
okt = Okt()
def process_morph(sentence):
    return ' '.join(okt.morphs(sentence))

# 문장 전처리
def clean_and_morph(sentence, is_question=True):
    # 한글 문장 전처리
    sentence = clean_sentence(sentence)
    # 형태소 변환
    sentence = process_morph(sentence)
    # Question 인 경우, Answer인 경우를 분기하여 처리합니다.
    if is_question:
        return sentence
    else:
        # START 토큰은 decoder input에 END 토큰은 decoder output에 추가합니다.
        return ('<START> ' + sentence, sentence + ' <END>')

def preprocess(texts, pairs):
    questions = []
    answer_in = []
    answer_out = []

    # 질의에 대한 전처리
    for text in texts:
        # 전처리와 morph 수행
        question = clean_and_morph(text, is_question=True)
        questions.append(question)

    # 답변에 대한 전처리
    for pair in pairs:
        # 전처리와 morph 수행
        in_, out_ = clean_and_morph(pair, is_question=False)
        answer_in.append(in_)
        answer_out.append(out_)
    
    return questions, answer_in, answer_out

questions, answer_in, answer_out = preprocess(texts, pairs)
# print(questions[:2])
# print(answer_in[:2])
# print(answer_out[:2])

all_sentences = questions + answer_in + answer_out

# 라이브러리 불어오기
import numpy as np
import warnings
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# WARNING 무시
warnings.filterwarnings('ignore')

tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')
tokenizer.fit_on_texts(all_sentences)

# 치환: 텍스트를 시퀀스로 인코딩 (texts_to_sequences)
question_sequence = tokenizer.texts_to_sequences(questions)
answer_in_sequence = tokenizer.texts_to_sequences(answer_in)
answer_out_sequence = tokenizer.texts_to_sequences(answer_out)

# 문장의 길이 맞추기 (pad_sequences)
MAX_LENGTH = 30
question_padded = pad_sequences(question_sequence, 
                                maxlen=MAX_LENGTH, 
                                truncating='post', 
                                padding='post')
answer_in_padded = pad_sequences(answer_in_sequence, 
                                 maxlen=MAX_LENGTH, 
                                 truncating='post', 
                                 padding='post')
answer_out_padded = pad_sequences(answer_out_sequence, 
                                  maxlen=MAX_LENGTH, 
                                  truncating='post', 
                                  padding='post')

# print(question_padded.shape, answer_in_padded.shape, answer_out_padded.shape)

# 라이브러리 로드
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

class Encoder(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_dim, time_steps):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, 
                                   embedding_dim, 
                                   input_length=time_steps)
        self.dropout = Dropout(0.2)
        self.lstm = LSTM(units, return_state=True)
        
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout(x)
        x, hidden_state, cell_state = self.lstm(x)
        return [hidden_state, cell_state]

# 디코더
class Decoder(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_dim, time_steps):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, 
                                   embedding_dim, 
                                   input_length=time_steps)
        self.dropout = Dropout(0.2)
        self.lstm = LSTM(units, 
                         return_state=True, 
                         return_sequences=True, 
                        )
        self.dense = Dense(vocab_size, activation='softmax')
    
    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        x = self.dropout(x)
        x, hidden_state, cell_state = self.lstm(x, initial_state=initial_state)        
        x = self.dense(x)
        return x, hidden_state, cell_state

# 모델 결합
class Seq2Seq(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_dim, time_steps, start_token, end_token):
        super(Seq2Seq, self).__init__()
        self.start_token = start_token
        self.end_token = end_token
        self.time_steps = time_steps
        
        self.encoder = Encoder(units, vocab_size, embedding_dim, time_steps)
        self.decoder = Decoder(units, vocab_size, embedding_dim, time_steps)
        
    def call(self, inputs, training=True):
        if training:
            encoder_inputs, decoder_inputs = inputs
            context_vector = self.encoder(encoder_inputs)
            decoder_outputs, _, _ = self.decoder(inputs=decoder_inputs, 
                                                 initial_state=context_vector)
            return decoder_outputs
        else:
            context_vector = self.encoder(inputs)
            target_seq = tf.constant([[self.start_token]], dtype=tf.float32)
            results = tf.TensorArray(tf.int32, self.time_steps)
            
            for i in tf.range(self.time_steps):
                decoder_output, decoder_hidden, decoder_cell = self.decoder(target_seq, 
                                                                            initial_state=context_vector)
                decoder_output = tf.cast(tf.argmax(decoder_output, axis=-1), 
                                         dtype=tf.int32)
                decoder_output = tf.reshape(decoder_output, shape=(1, 1))
                results = results.write(i, decoder_output)
                
                if decoder_output == self.end_token:
                    break
                    
                target_seq = decoder_output
                context_vector = [decoder_hidden, decoder_cell]
                
            return tf.reshape(results.stack(), shape=(1, self.time_steps))

VOCAB_SIZE = len(tokenizer.word_index)+1

def convert_to_one_hot(padded):
    # 원핫인코딩 초기화
    one_hot_vector = np.zeros((len(answer_out_padded), 
                               MAX_LENGTH, 
                               VOCAB_SIZE))

    # 디코더 목표를 원핫인코딩으로 변환
    # 학습시 입력은 인덱스이지만, 출력은 원핫인코딩 형식임
    for i, sequence in enumerate(answer_out_padded):
        for j, index in enumerate(sequence):
            one_hot_vector[i, j, index] = 1

    return one_hot_vector

answer_in_one_hot = convert_to_one_hot(answer_in_padded)
answer_out_one_hot = convert_to_one_hot(answer_out_padded)
answer_in_one_hot[0].shape, answer_in_one_hot[0].shape

# 변환된 index를 다시 단어로 변환
def convert_index_to_text(indexs, end_token): 
    
    sentence = ''
    
    # 모든 문장에 대해서 반복
    for index in indexs:
        if index == end_token:
            # 끝 단어이므로 예측 중비
            break;
        # 사전에 존재하는 단어의 경우 단어 추가
        if index > 0 and tokenizer.index_word[index] is not None:
            sentence += tokenizer.index_word[index]
        else:
        # 사전에 없는 인덱스면 빈 문자열 추가
            sentence += ''
            
        # 빈칸 추가
        sentence += ' '
    return sentence

# 하이퍼 파라미터 정의
BUFFER_SIZE = 60
BATCH_SIZE = 16
EMBEDDING_DIM = 100
TIME_STEPS = MAX_LENGTH
START_TOKEN = tokenizer.word_index['<START>']
END_TOKEN = tokenizer.word_index['<END>']

UNITS = 128

VOCAB_SIZE = len(tokenizer.word_index)+1
DATA_LENGTH = len(questions)
SAMPLE_SIZE = 3
NUM_EPOCHS = 10

# 체크포인트 생성
checkpoint_path = 'model/seq2seq-chatbot-no-attention-checkpoint.ckpt'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                             save_weights_only=True,
                             save_best_only=True, 
                             monitor='loss', 
                             verbose=1
                            )
                            
# seq2seq
seq2seq = Seq2Seq(UNITS, 
                  VOCAB_SIZE, 
                  EMBEDDING_DIM, 
                  TIME_STEPS, 
                  START_TOKEN, 
                  END_TOKEN)

seq2seq.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['acc'])
def make_prediction(model, question_inputs):
    results = model(inputs=question_inputs, training=False)
    # 변환된 인덱스를 문장으로 변환
    results = np.asarray(results).reshape(-1)
    return results
for epoch in range(NUM_EPOCHS):
    print(f'processing epoch: {epoch * 10 + 1}...')
    seq2seq.fit([question_padded, answer_in_padded],
                answer_out_one_hot,
                epochs=10,
                batch_size=BATCH_SIZE,
                callbacks=[checkpoint]
               )
    # 랜덤한 샘플 번호 추출
    samples = np.random.randint(DATA_LENGTH, size=SAMPLE_SIZE)

    # 예측 성능 테스트
    for idx in samples:
        question_inputs = question_padded[idx]
        # 문장 예측
        results = make_prediction(seq2seq, np.expand_dims(question_inputs, 0))
        
        # 변환된 인덱스를 문장으로 변환
        results = convert_index_to_text(results, END_TOKEN)
        
        print(f'Q: {questions[idx]}')
        print(f'A: {results}\n')
        print()


# 챗봇

def make_question(sentence):
    sentence = clean_and_morph(sentence)
    question_sequence = tokenizer.texts_to_sequences([sentence])
    question_padded = pad_sequences(question_sequence, maxlen=MAX_LENGTH, truncating='post', padding='post')
    return question_padded

#make_question('오늘 날씨 어때?')
    
def run_chatbot(question):
    question_inputs = make_question(question)
    results = make_prediction(seq2seq, question_inputs)
    results = convert_index_to_text(results, END_TOKEN)
    return results

# 챗봇 실행
while True:
    user_input = input('<< 말을 걸어 보세요!\n')
    if user_input == 'q':
        break
    print('>> 챗봇 응답: {}'.format(run_chatbot(user_input)))