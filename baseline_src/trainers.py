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


##### SYMBOLIC MODEL WITH UNET INPUT TRAINER #######

class SymbolicAttentionTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, config):
        self.model = model.to(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.config = config
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = self.config.get("num_epochs", 100)
        self.patience = self.config.get("patience", 10)
        self.save_path = self.config.get("save_path", "symbolic_model_best.pth")

        self.best_val_loss = float("inf")
        self.wait = 0
        self.best_epoch = 0

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_train_loss = 0.0

            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for batch in loop:
                x = batch.to(self.device)

                self.optimizer.zero_grad()
                x_recon, symbols = self.model(x, hard=False)

                sym_b = symbols[0]
                sym_skip = symbols[1]

                loss = self.loss_fn(x_recon, x)

                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_train_loss = total_train_loss / len(self.train_loader)
            avg_val_loss = self._validate()

            print(
                f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.6f}  |  Val Loss: {avg_val_loss:.6f} | Patience: {self.wait}")

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_epoch = epoch
                self.wait = 0
                self._save_checkpoint(epoch)
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"🛑 Early stopping at epoch {epoch + 1} (best epoch was {self.best_epoch + 1})")
                    break

    def _validate(self):
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch.to(self.device)
                x_recon, _ = self.model(x, hard=True)
                loss = self.loss_fn(x_recon, x)
                total_val_loss += loss.item()
        return total_val_loss / len(self.val_loader)

    def _save_checkpoint(self, epoch):
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }, self.save_path)


##### QNA TRAINER SYMBOLIC MODEL IS INPUT #####

class QnATrainer:
    def __init__(self, answer_model, symbolic_model, train_loader, val_loader,
                 train_dataset, val_dataset, criterion, optimizer, config):

        self.answer_model = answer_model
        self.symbolic_model = symbolic_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.criterion = criterion
        self.optimizer = optimizer

        # Config hyperparameters
        self.num_epochs = config.get("num_epochs", 20)
        self.patience = config.get("patience", 5)
        self.save_path = config.get("save_path", "best_similarity_model.pth")
        self.device = config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')

        self.answer_model.to(self.device)
        self.symbolic_model.to(self.device)

        # Early stopping state
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.best_model_state = None

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self._validate(epoch)

            print(f"\nEpoch {epoch + 1}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.answer_model.state_dict()
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print("🛑 Early stopping triggered.")
                break

        if self.best_model_state is not None:
            self.answer_model.load_state_dict(self.best_model_state)
            print("✅ Best model weights restored.")
            torch.save(self.best_model_state, self.save_path)

    def _train_one_epoch(self, epoch):
        self.answer_model.train()
        total_loss, correct = 0.0, 0

        for img, questions, answer in tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}"):
            img, questions, answer = img.to(self.device), questions.to(self.device), answer.to(self.device)
            img_recon, symbols = self.symbolic_model(img, hard=True)

            logits = self.answer_model(img_recon, questions)
            loss = self.criterion(logits, answer)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == answer).sum().item()

        train_acc = correct / len(self.train_dataset)
        return total_loss, train_acc

    def _validate(self, epoch):
        self.answer_model.eval()
        total_loss, correct = 0.0, 0

        with torch.no_grad():
            for img, questions, answer in self.val_loader:
                img, questions, answer = img.to(self.device), questions.to(self.device), answer.to(self.device)
                img_recon, symbols = self.symbolic_model(img, hard=True)

                logits = self.answer_model(img_recon, questions)
                loss = self.criterion(logits, answer)

                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == answer).sum().item()

        val_acc = correct / len(self.val_dataset)
        return total_loss, val_acc


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

            symbols = self.sender(x, hard=False)
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
