import torch
import torch.nn as nn
from tqdm import tqdm


##### UNET PRETRAINER #####

class UNetTrainer:
    def __init__(self, model, dataloader_train, dataloader_val, config):
        self.model = model.to(config['device'])
        self.train_loader = dataloader_train
        self.val_loader = dataloader_val

        # Extract config
        self.device = config.get('device', 'cuda')
        self.patience = config.get('patience', 5)
        self.checkpoint_path = config.get('checkpoint_path', 'unet_best.pth')
        self.loss_fn = config.get('loss_fn', nn.MSELoss())
        self.optimizer = config.get('optimizer', torch.optim.Adam(
            model.parameters(), lr=config.get('lr', 1e-3)
        ))

        self.best_val_loss = float('inf')
        self.wait = 0
        self.best_epoch = 0

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self._train_epoch(epoch, num_epochs)
            val_loss = self._validate_epoch()

            print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.6f}  |  Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.wait = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"✅ Saved new best model at epoch {epoch + 1}")
            else:
                self.wait += 1
                print(f"⚠️ No improvement. Patience: {self.wait}/{self.patience}")
                if self.wait >= self.patience:
                    print(f"🛑 Early stopping at epoch {epoch + 1} (best was epoch {self.best_epoch + 1})")
                    break

    def _train_epoch(self, epoch, num_epochs):
        self.model.train()
        total_loss = 0.0
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in loop:
            x = batch.to(self.device)
            self.optimizer.zero_grad()

            recon = self.model(x)
            loss = self.loss_fn(recon, x)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch.to(self.device)
                recon = self.model(x)
                loss = self.loss_fn(recon, x)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)


##### BASELINE SYMBOLIC COMPRESSION TRAINER #####

class SymbolicTrainer:
    def __init__(self, sender, receiver, optimizer, loss_fn, dataloader_train, dataloader_val, config):
        self.sender = sender.to(config['device'])
        self.receiver = receiver.to(config['device'])
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = dataloader_train
        self.val_loader = dataloader_val

        # Config
        self.device = config.get('device', 'cuda')
        self.patience = config.get('patience', 5)
        self.checkpoint_path = config.get('checkpoint_path', 'symbolic_best.pth')

        # Early stopping state
        self.best_val_loss = float('inf')
        self.wait = 0
        self.best_epoch = 0

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self._train_epoch(epoch, num_epochs)
            val_loss = self._validate_epoch(epoch, num_epochs)

            print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.wait = 0
                self._save_checkpoint(epoch)
                print(f"✅ Saved new best model at epoch {epoch + 1}")
            else:
                self.wait += 1
                print(f"⚠️ No improvement. Patience: {self.wait}/{self.patience}")
                if self.wait >= self.patience:
                    print(f"🛑 Early stopping triggered at epoch {epoch + 1} (best was epoch {self.best_epoch + 1})")
                    break

    def _train_epoch(self, epoch, num_epochs):
        self.sender.train()
        self.receiver.train()
        total_loss = 0.0

        loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for batch in loop:
            x = batch.to(self.device)
            self.optimizer.zero_grad()

            symbols = self.sender(x, hard=True)
            x_recon = self.receiver(symbols)
            loss = self.loss_fn(x_recon, x)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def _validate_epoch(self, epoch, num_epochs):
        self.sender.eval()
        self.receiver.eval()
        total_loss = 0.0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [ Val ]")
            for batch in loop:
                x = batch.to(self.device)
                symbols = self.sender(x, hard=True)
                x_recon = self.receiver(symbols)
                loss = self.loss_fn(x_recon, x)
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

        return total_loss / len(self.val_loader)

    def _save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'sender_state': self.sender.state_dict(),
            'receiver_state': self.receiver.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }, self.checkpoint_path)
