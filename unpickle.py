import pickle



with open("length_influences1_dict.pkl", 'rb') as f:
    influences = pickle.load(f)

print(influences)