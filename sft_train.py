# =======================================================================
#                                  SFT Training 
# =======================================================================

import torch 
import torch.nn as nn 


def estimate_loss(model, train_loader, val_loader, eval_iters, device):
    result = {}
    model.eval()

    for split, loader in [("train", train_loader), ("val", val_loader)]:
         losses = []

         for i, (x,y,loss_mask) in enumerate(loader):
              if i >eval_iters:
                   break 
              x = x.to(device)
              y = y.to(device)
              loss_mask = loss_mask.to(device)

              logits, loss = model(x,y, loss_mask)
              losses.append(loss)
        
         result[split] = sum(losses)/len(losses)
    
    return result

def train(model, train_loader, val_loader, eval_interval, eval_iters, n_epochs, learning_rate, device):
     optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate)

     model.to(device)
     model.train()

     for epoch in range(n_epochs):
         for batch_idx, (x,y,loss_mask) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            loss_mask = loss_mask.to(device)

            logits, loss = model(x,y, loss_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
         if epoch % eval_interval ==0 : 
             losses = estimate_loss(model, train_loader, val_loader, eval_iters, device)

             print(
                f"epoch {epoch}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

