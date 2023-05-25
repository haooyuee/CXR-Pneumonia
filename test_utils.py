import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

def test_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    y_pred_p = []

    with torch.no_grad():
        for batch in tqdm(dataloader['test']):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_p.extend(torch.softmax(outputs, dim=1).cpu().numpy()[:, 1])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_p = np.array(y_pred_p)
    #print('y_true',y_true)
    #print('y_true',y_pred)
    #print('y_true',y_pred_p)


    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_p)

    return accuracy, f1, auc
