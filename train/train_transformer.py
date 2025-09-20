import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib, os
from models.transformer_model import TransformerAutoencoder
from sklearn.preprocessing import StandardScaler

def create_sequences(data, seq_len=20):
    xs = []
    for i in range(len(data) - seq_len):
        xs.append(data[i:(i+seq_len)])
    return np.array(xs)

def main(data_path="data/geolife_features.csv", epochs=30, seq_len=20, batch_size=128, lr=1e-3):
    df = pd.read_csv(data_path)
    features = ["dt", "dist", "speed", "bearing", "d_bearing", "accel"]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].values)

    seqs = create_sequences(X, seq_len)
    X_tensor = torch.tensor(seqs, dtype=torch.float32)

    model = TransformerAutoencoder(seq_len, len(features))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            X_batch = batch[0]
            recon = model(X_batch)
            loss = criterion(recon, X_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        print(f"Epoch {epoch+1}/{epochs}, Loss={epoch_loss/len(X_tensor):.6f}")

    os.makedirs("models/artifacts", exist_ok=True)
    torch.save(model.state_dict(), "models/artifacts/transformer_ae_geolife.pt")
    joblib.dump(scaler, "models/artifacts/scaler_geolife.pkl")
    print("✅ Saved model and scaler.")

if __name__ == "__main__":
    main()
