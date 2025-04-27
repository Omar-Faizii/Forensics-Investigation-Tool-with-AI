import re
import pandas as pd
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# === Step 1: Load and Prepare Dataset ===
print("Loading dataset...")
data = pd.read_csv(r'C:\Users\Administrator\Desktop\DF PROJECT\zahhak\data.csv', encoding='utf-8')  # Load the uploaded CSV file
data = data.dropna(subset=['Query'])  # Ensure no null values in the 'Query' column
queries = data['Query']  # Extract the log queries
labels = data['Label']  # Extract the labels

# Preprocess queries: Convert text to numerical vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(queries).toarray()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# === Step 2: Define the Model ===
class LogClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):  # Fixed __init__
        super(LogClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Model parameters
input_size = X_train.shape[1]
hidden_size = 128
num_classes = len(label_encoder.classes_)
model = LogClassifier(input_size, hidden_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Step 3: Train the Model ===
print("Training the model...")
epochs = 30
for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'forensic_model.pth')
print("Model training complete. Model saved as 'forensic_model.pth'.")

# === Step 4: Forensic Tool with the Trained Model ===
print("Initializing forensic tool...")

# Load the trained model
model.load_state_dict(torch.load('forensic_model.pth'))
model.eval()

# Function to parse Apache log format
def parse_apache_log(log_entry):
    """
    Extracts the request (query string) from an Apache log entry in Common Log Format (CLF).
    """
    # log_pattern = r'\"(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH)\s+([^\s]+)\s+HTTP/\d\.\d\"'
    log_pattern = r'\"(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH)\s+([^\"]+)\s+HTTP/\d\.\d\"'
    match = re.search(log_pattern, log_entry)
    if match:
        return match.group(2)  # Extract the request path/query
    return None

# Function to analyze a single log entry
def analyze_log(log_entry):
    # Parse the Apache log to extract the query/request
    query = parse_apache_log(log_entry)
    if query is None:
        return "Unable to parse log entry"

    # Vectorize the query
    log_vector = vectorizer.transform([query]).toarray()
    log_tensor = torch.tensor(log_vector, dtype=torch.float32)

    # Predict using the trained model
    with torch.no_grad():
        output = model(log_tensor)
        prediction = torch.argmax(output, dim=1).item()
        prediction_label = label_encoder.inverse_transform([prediction])[0]

    return prediction_label

# === Step 5: Analyze Logs from log.txt ===
print("Analyzing logs from log.txt...")

# Path to the log.txt file
log_file_path = r'C:\Users\Administrator\Desktop\DF PROJECT\zahhak\log.txt'

# Open and read the log file
with open(log_file_path, 'r', encoding='utf-8') as log_file:
    log_entries = log_file.readlines()

# Analyze each log entry
results = []
for log_entry in log_entries:
    log_entry = log_entry.strip()  # Remove any leading/trailing whitespace
    if log_entry:  # Skip empty lines
        prediction = analyze_log(log_entry)
        results.append((log_entry, prediction))

# Print the results
print("\nAnalysis Results:")
for log_entry, prediction in results:
    print(f"Log Entry: {log_entry}\nPrediction: {prediction}\n")

# Optionally save results to a file
output_file = 'analysis_results.txt'
with open(output_file, 'w', encoding='utf-8') as result_file:
    for log_entry, prediction in results:
        result_file.write(f"Log Entry: {log_entry}\nPrediction: {prediction}\n\n")

print(f"Analysis complete. Results saved to {output_file}.")