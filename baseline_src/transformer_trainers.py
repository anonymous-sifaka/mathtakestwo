import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

##### MLLM QNA TRAINER (VISION + SYMBOLIC TRANSFORMER) #####

class MLLMQnATrainer:
    def __init__(self, mllm_model, train_loader, val_loader,
                 train_dataset, val_dataset, criterion, optimizer, config):

        self.model = mllm_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.criterion = criterion
        self.optimizer = optimizer

        # Config hyperparameters
        self.num_epochs = config.get("num_epochs", 20)
        self.patience = config.get("patience", 5)
        self.save_path = config.get("save_path", "best_mllm_model.pth")
        self.device = config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

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
                self.best_model_state = self.model.state_dict()
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print("🛑 Early stopping triggered.")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("✅ Best model weights restored.")
            torch.save(self.best_model_state, self.save_path)

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss, correct = 0.0, 0
        total_samples = 0

        for batch in tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}"):
            # Handle different data structures from dataloaders
            if len(batch) == 3:  # img, questions, answer
                img, questions, answer = batch
            elif len(batch) == 5:  # img, questions, answer, program, options
                img, questions, answer, program, options = batch
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")
            
            img, questions, answer = img.to(self.device), questions.to(self.device), answer.to(self.device)

            logits = self.model(img, questions)
            loss = self.criterion(logits, answer)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == answer).sum().item()
            total_samples += answer.size(0)

        train_acc = correct / total_samples if total_samples > 0 else 0.0
        return total_loss, train_acc

    def _validate(self, epoch):
        self.model.eval()
        total_loss, correct = 0.0, 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Handle different data structures from dataloaders
                if len(batch) == 3:  # img, questions, answer
                    img, questions, answer = batch
                elif len(batch) == 5:  # img, questions, answer, program, options
                    img, questions, answer, program, options = batch
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")

                img, questions, answer = img.to(self.device), questions.to(self.device), answer.to(self.device)

                logits = self.model(img, questions)
                loss = self.criterion(logits, answer)

                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == answer).sum().item()
                total_samples += answer.size(0)

        val_acc = correct / total_samples if total_samples > 0 else 0.0
        return total_loss, val_acc

##### MLLM RECONSTRUCTION TRAINER #####

class MLLMReconstructionTrainer:
    def __init__(self, reconstruction_model, train_loader, val_loader,
                 train_dataset, val_dataset, criterion, optimizer, config):

        self.model = reconstruction_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.criterion = criterion
        self.optimizer = optimizer

        # Config hyperparameters
        self.num_epochs = config.get("num_epochs", 20)
        self.patience = config.get("patience", 5)
        self.save_path = config.get("save_path", "best_reconstruction_model.pth")
        self.device = config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

        # Early stopping state
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_model_state = None

    def transform_image(self, image):
        """Transform image to 224x224 RGB format for Vision Transformer"""
        if image.dim() == 4:  # Batch of images
            transformed_images = []
            for i in range(image.size(0)):
                img = image[i]
                if img.dim() == 3 and img.size(0) == 1:  # Single channel
                    img = img.repeat(3, 1, 1)  # Convert to RGB
                elif img.dim() == 2:  # 2D image
                    img = img.unsqueeze(0).repeat(3, 1, 1)  # Convert to RGB
                
                # Resize to 224x224
                img = F.interpolate(
                    img.unsqueeze(0), 
                    size=(224, 224), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                # Normalize
                img = (img - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img.device)) / \
                      torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img.device)
                
                transformed_images.append(img)
            return torch.stack(transformed_images)
        else:
            # Single image
            if image.dim() == 3 and image.size(0) == 1:  # Single channel
                image = image.repeat(3, 1, 1)  # Convert to RGB
            elif image.dim() == 2:  # 2D image
                image = image.unsqueeze(0).repeat(3, 1, 1)  # Convert to RGB
            
            # Resize to 224x224
            image = F.interpolate(
                image.unsqueeze(0), 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            # Normalize
            image = (image - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)) / \
                    torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)
            
            return image

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self._train_one_epoch(epoch)
            val_loss = self._validate(epoch)

            print(f"\nEpoch {epoch + 1}: "
                  f"Train Loss={train_loss:.4f} | "
                  f"Val Loss={val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print("🛑 Early stopping triggered.")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("✅ Best model weights restored.")
            torch.save(self.best_model_state, self.save_path)

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}"):
            # Handle different data structures from dataloaders
            if len(batch) == 3:  # img, questions, answer
                img, questions, answer = batch
            elif len(batch) == 5:  # img, questions, answer, program, options
                img, questions, answer, program, options = batch
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")
            
            img = img.to(self.device)
            
            # Transform images to 224x224 RGB format
            img = self.transform_image(img)

            # Forward pass - reconstruct the image
            reconstructed, symbols, msg_embed = self.model(img)
            
            # Compute reconstruction loss (MSE between original and reconstructed)
            loss = self.criterion(reconstructed, img)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate(self, epoch):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                # Handle different data structures from dataloaders
                if len(batch) == 3:  # img, questions, answer
                    img, questions, answer = batch
                elif len(batch) == 5:  # img, questions, answer, program, options
                    img, questions, answer, program, options = batch
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")

                img = img.to(self.device)
                
                # Transform images to 224x224 RGB format
                img = self.transform_image(img)

                # Forward pass - reconstruct the image
                reconstructed, symbols, msg_embed = self.model(img)
                
                # Compute reconstruction loss
                loss = self.criterion(reconstructed, img)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)
