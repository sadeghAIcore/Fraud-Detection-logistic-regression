
import torch 
from model import FraudClassifierMLP
from datasets import load_data


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def accuracy_fn(y_true , y_pred ):
    correct = torch.eq(y_true , y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
loss_fn = nn.BCEWithLogitsLoss()

train_loader , test_loader , input_dim = load_data() 
model = FraudClassifierMLP(input_dim=input_dim).to(device)

model.load_state_dict(torch.load('../models/model.pth', map_location=device))

def predict(test_dataloader=test_loader):
    model.eval()
    test_acc_running = 0.0
    test_loss_running = 0.0

    with torch.no_grad():
        for batch, (X_test, y_test) in enumerate(test_dataloader):
            X_test = X_test.to(device)
            y_test = y_test.float().to(device).unsqueeze(1)

            y_logits = model(X_test)
            y_pred = torch.round(torch.sigmoid(y_logits))

            loss = loss_fn(y_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=y_pred)

            test_acc_running += test_acc
            test_loss_running += loss.item()

    final_test_acc = test_acc_running / len(test_dataloader)
    final_test_loss = test_loss_running / len(test_dataloader)

    print(f'test acc : {final_test_acc} | test loss : {final_test_loss}')
