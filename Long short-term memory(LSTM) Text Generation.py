"""
Generate text using a Long Short-Term Memory (LSTM) model.
"""

# %% importing libraries
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter # word frequency count
from itertools import product # combinations of two iterables

# %%  data loading and preprocessing
text = """Once upon a time, in a quiet village surrounded by hills,
there lived a young girl named Elara. She loved to explore the forest, 
listen to the birds, and watch the stars at night. 
Every evening, she would sit by the old oak tree and write stories in her little notebook.
Her imagination knew no bounds. One day, she found a mysterious map hidden under a rock near the river.
That map led her on an adventure she would never forget."""

words = text.replace('\n', ' ').replace('.', ' ').lower().split() # remove new lines and periods, convert to lowercase, and split into words
word_count = Counter(words) # count the frequency of each word
vocab = sorted(word_count, key=word_count.get, reverse=True) # sort words by frequency

word_to_ix = {word: i for i, word in enumerate(vocab)} # create a mapping from words to indices
ix_to_word = {i: word for i, word in enumerate(vocab)} # create a mapping from indices to words

# prepare the data for training
data = [(words[i], words[i + 1]) for i in range(len(words) - 1)] # create pairs of (word, next word)

# %% define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # embedding layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True) # LSTM layer
        self.fc = nn.Linear(hidden_dim, vocab_size) # fully connected layer

    def forward(self, x):
        x = self.embedding(x).unsqueeze(0) # get the embeddings
        lstm_out, _ = self.lstm(x) # get the output from LSTM layer
        output = self.fc(lstm_out[:, -1, :]) # get the output from the last time step
        return output

model = LSTMModel(len(vocab), embedding_dim=8, hidden_dim=32) # create the model

 
# %% hyperparameters tuning
# word list to tensor mapping
def prepare_sequence(seq, to_ix):
    return torch.tensor([to_ix[w] for w in seq], dtype=torch.long) # convert words to indices

# Define hyperparameters tuning combinations
embedding_sizes = [8, 16] # different embedding sizes
hidden_sizes = [32, 64] # different hidden sizes
learning_rates = [0.01, 0.005] # different learning rates

best_loss = float('inf') # initialize best loss to infinity to find the minimum loss
best_params = {} # Empty dictionary for save the best parameters

print("Tuning hyperparameters...")
for embedding_size, hidden_size, learning_rate in product(embedding_sizes, hidden_sizes, learning_rates):
    model = LSTMModel(len(vocab), embedding_dim=embedding_size, hidden_dim=hidden_size) # create the model with current hyperparameters
    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # optimizer

    # Training loop
    for epoch in range(50): # 50 epochs
        total_loss = 0
        for word, next_word in data:
            model.zero_grad() # zero the gradients
            word_tensor = prepare_sequence([word], word_to_ix) # convert word to tensor
            next_word_tensor = torch.tensor([word_to_ix[next_word]], dtype=torch.long) # convert next word to tensor

            output = model(word_tensor) # get the output from the model
            loss = criterion(output, next_word_tensor) # calculate the loss
            loss.backward() # backpropagation
            optimizer.step() # update the weights

            total_loss += loss.item() # accumulate the loss

        avg_loss = total_loss / len(data) # average loss for the epoch

        if epoch % 10 == 0:
            print(f"Embedding size: {embedding_size}, Hidden size: {hidden_size}, Learning rate: {learning_rate}, Epoch: {epoch}, Loss: {avg_loss:.4f}")
            
    if avg_loss < best_loss: # check if the current loss is less than the best loss
        best_loss = avg_loss # update the best loss
        best_params = {
            'embedding_size': embedding_size,
            'hidden_size': hidden_size,
            'learning_rate': learning_rate
        } # update the best parameters
        
print(f"New best parameters found: {best_params} with loss: {best_loss:.4f}")
            
# %% lstm training

final_model = LSTMModel(len(vocab), embedding_dim=best_params['embedding_size'], hidden_dim=best_params['hidden_size']) # create the final model with best parameters
optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate']) # optimizer
criterion = nn.CrossEntropyLoss() # loss function

print("Training the final model...")
for epoch in range(100): # 100 epochs
    total_loss = 0
    for word, next_word in data:
        final_model.zero_grad() # zero the gradients
        input_tensor = prepare_sequence([word], word_to_ix) # convert word to tensor
        target_tensor = torch.tensor([word_to_ix[next_word]], dtype=torch.long)
        output = final_model(input_tensor)
        loss = criterion(output, target_tensor) # calculate the loss
        loss.backward() # backpropagation
        optimizer.step() # update the weights
        total_loss += loss.item() # accumulate the loss
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Total Loss: {total_loss:.4f}")
        
# %% test
# Predict the next word
def predict_sequence(start_word, num_words):
    current_word = start_word
    output_words = [current_word]
    for _ in range(num_words): 
        with torch.no_grad():
              input_tensor = prepare_sequence([current_word], word_to_ix)
              output = final_model(input_tensor)
              predicted_index = torch.argmax(output).item()
              predicted_word = ix_to_word[predicted_index]
              output_words.append(predicted_word)
              current_word = predicted_word
    return output_words

start_word = "elara" # starting word for prediction
num_words = 10 # number of words to predict
predicted_sequence = predict_sequence(start_word, num_words) # predict the sequence
print("Predicted sequence:", ' '.join(predicted_sequence)) # print the predicted sequence