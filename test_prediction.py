"""
Test script to verify the model works without interactive input
"""
import pickle

# Load the model
print("Loading model...")
load_model = pickle.load(open('final_model.sav', 'rb'))

# Test statements
test_statements = [
    "Mexico will pay for the wall",
    "The president announced new economic policies today",
    "Scientists discover cure for cancer"
]

print("\nTesting predictions:\n")
for statement in test_statements:
    prediction = load_model.predict([statement])
    probability = load_model.predict_proba([statement])

    print(f"Statement: {statement}")
    print(f"Prediction: {prediction[0]}")
    print(f"Probability: {max(probability[0])*100:.2f}%")
    print("-" * 60)

print("\nModel is working correctly!")
