from train.nlp_disaster_classifier import NLPDisasterClassifier


try:

    train = NLPDisasterClassifier()
    train.train()

except Exception as e:
    print(e)
