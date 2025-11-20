import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('tokenizers/amharic_extended_bpe.model')
print(f'Tokenizer vocab size: {sp.vocab_size()}')

test = 'ሰላም ልጆች እንዴት ናችሁ'
tokens = sp.encode(test)
print(f'Test text: {test}')
print(f'Tokens: {tokens}')
print(f'Token range: {min(tokens)}-{max(tokens)}')
