import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class ChurnDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class ChurnTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, dim_feedforward=128, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=num_heads,
            dim_feedforward=dim_feedforward * 2,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(dim_feedforward, 2)  # 2 classes: churned or not churned
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

def prepare_data(df1, df2):
    """
    Prepare data from two time periods for churn detection
    """
    # Identify churners
    clients_period1 = set(df1['client_id'].unique())
    clients_period2 = set(df2['client_id'].unique())
    
    churners = clients_period1 - clients_period2
    
    # Create features from period 1 data
    features = []
    labels = []
    
    for client_id in clients_period1:
        client_data = df1[df1['client_id'] == client_id]
        
        # Calculate features
        feature_dict = {
            'total_revenue': client_data['revenue'].sum(),
            'avg_transaction_value': client_data['revenue'].mean(),
            'transaction_count': len(client_data),
            'unique_products': len(client_data['product_id'].unique()),
            'product_categories': len(client_data['product_category'].unique()),
            'avg_quantity_per_transaction': client_data['quantity'].mean(),
            'total_loss': client_data['loss'].sum(),
            'days_active': len(client_data['transaction_date'].unique())
        }
        
        # Add product category distribution
        category_counts = client_data['product_category'].value_counts(normalize=True)
        for cat in df1['product_category'].unique():
            feature_dict[f'category_ratio_{cat}'] = category_counts.get(cat, 0)
            
        features.append(list(feature_dict.values()))
        labels.append(1 if client_id in churners else 0)
    
    return np.array(features), np.array(labels)

def train_churn_model(df1, df2, epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train the transformer model for churn prediction
    """
    # Prepare data
    features, labels = prepare_data(df1, df2)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features_scaled, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = ChurnDataset(X_train, y_train)
    val_dataset = ChurnDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = features.shape[1]
    model = ChurnTransformer(input_dim)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        val_accuracy = 100 * correct / total
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%')
    
    # Load best model
    model.load_state_dict(best_model)
    return model, scaler

def predict_churn(model, scaler, client_data):
    """
    Predict churn probability for new client data
    """
    model.eval()
    
    # Prepare features (similar to prepare_data function)
    features = []
    # ... (feature preparation code similar to prepare_data function)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features_scaled)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
    return probabilities[:, 1].numpy()  # Return churn probabilities

# Example usage:
"""
# Load your data
df_period1 = pd.read_csv('period1_data.csv')
df_period2 = pd.read_csv('period2_data.csv')

# Train the model
model, scaler = train_churn_model(df_period1, df_period2)

# Make predictions for new data
new_client_data = prepare_new_client_data()  # Your data preparation function
churn_probabilities = predict_churn(model, scaler, new_client_data)
"""