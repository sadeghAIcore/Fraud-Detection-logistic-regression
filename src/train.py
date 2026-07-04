## Now were going to train our model 
import torch 
from torch import nn 
from model import FraudClassifierMLP
from datasets import load_data


def accuracy_fn(y_true , y_pred ):
    correct = torch.eq(y_true , y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100

    return acc

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader , test_loader , input_dim = load_data() 

model = FraudClassifierMLP(input_dim=input_dim).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters() , lr= 0.01)

def train_model(epochs : int = 20 ,train_dataloader = train_loader ,loss_fn = loss_fn ,optimizer = optimizer  ,
                device = device  , accuracy_fn = accuracy_fn):

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        running_acc = 0.0

        for batch , (X_batch , y_batch) in enumerate(train_dataloader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_batch = y_batch.unsqueeze(1)

            y_pred_logits = model(X_batch)
            y_pred = torch.round(torch.sigmoid(y_pred_logits))

            loss = loss_fn(y_pred_logits , y_batch)
            acc = accuracy_fn(y_true= y_batch , y_pred = y_pred )

            running_loss += loss.item()
            running_acc += acc

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss = running_loss / len(train_dataloader)
            epoch_acc = running_acc / len(train_dataloader)


        if epoch % 10 == 0 :
            print(f"Epoch: {epoch+1:03d}/{epochs} | Loss: {epoch_loss:.5f} | Acc: {epoch_acc:.2f}%")
