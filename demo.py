import fasttext

# Load the model
model = fasttext.load_model('lid.176.ftz')
print(model.labels)

# Predict
print(model.predict('你好'))
print(model.predict('こんにちは'))
print(model.predict('hello'))
