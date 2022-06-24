from keras.models import model_from_json


def load_model():
    model = model_from_json(open('E:/paper_work/project/ser/media/save_model/model_architecture.json').read())
    model.load_weights('E:/paper_work/project/ser/media/save_model/best_weights.hdf5')
    return model


def predict(model, audio_feature):
    emotion_enc = {0: 'fear', 1: 'disgust', 2: 'neutral', 3: 'happy', 4: 'sadness', 5: 'surprise', 6: 'angry'}
    predictions = model.predict(audio_feature)
    output = predictions.argmax()
    return emotion_enc[output]
