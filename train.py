import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
import numpy as np

def train_model(model, train_loader, device, epochs=5):
    """モデルの学習"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epoch_losses = [] # ロスを保存するリスト
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss) # エポックごとのロスを保存
    return model,epoch_losses

def evaluate_model(model, test_loader, device, cycle, epoch=None):
    """モデルの評価(サンプルごとにロスを計算)"""
    criterion =  nn.CrossEntropyLoss(reduction='none')
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            #各サンプルのロスを計算
            losses = criterion(outputs, labels)
            
            #信頼度(各クラスの確率)を計算
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # CPU上のnumpy配列に変換（保存用）
            losses_np = losses.cpu().numpy()
            probs_np = probs.cpu().numpy()
            preds_np = predicted.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # 結果をリストに保存
            for i in range(len(labels)):
                row_data ={
                    'Cycle': cycle,
                    'Epoch': epoch if epoch is not None else 'Final',
                    'True Label': labels_np[i],
                    'Predicted' : preds_np[i],
                    'Loss': losses_np[i]
                }
                #各クラスの信頼度を追加
                for cls_idx in range(10):
                    row_data[f'Confidence_Class_{cls_idx}'] = probs_np[i][cls_idx]
                    
                results.append(row_data)
                
    #pandasのdataframeに変換
    df_results = pd.DataFrame(results)
    #全体の正解率を計算
    accuracy = (df_results['True Label'] == df_results['Predicted']).mean()
    return  df_results, accuracy