import matplotlib.pyplot as plt
import torch
import numpy as np


def visualize_reconstructions_unet(model, dataloader, num_examples=5, batch_size=1, title_prefix=""):
    """
    Visualize original vs. reconstructed images from a model and dataset.

    Args:
        model (torch.nn.Module): Trained model with .eval() mode and loaded weights.
        dataset (torch.utils.data.Dataset): Dataset object (e.g., PrecondDataset).
        num_examples (int): Number of examples to display.
        batch_size (int): Batch size for DataLoader.
        title_prefix (str): Optional title prefix (e.g., "Validation").
    """
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x = batch.cuda()  # Assumes shape [B, 1, H, W]
            recon = model(x)

            for b in range(x.size(0)):
                if i * batch_size + b >= num_examples:
                    return

                fig, axs = plt.subplots(1, 2, figsize=(6, 3))
                axs[0].imshow(x[b, 0].cpu().numpy(), cmap='gray')
                axs[0].set_title(f"{title_prefix} Original")
                axs[0].axis('off')

                axs[1].imshow(recon[b, 0].cpu().numpy(), cmap='gray')
                axs[1].set_title(f"{title_prefix} Reconstructed")
                axs[1].axis('off')

                plt.tight_layout()
                plt.show()


def visualize_img_reconstruction(symbolic_model, dataloader, device="cuda", num_examples=5):
    symbolic_model.eval()

    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)  # [B, 1, 47, 41]
            x_recon, symbols = symbolic_model(x, hard=True)

            x = x.cpu().numpy()
            x_recon = x_recon.cpu().numpy()

            symbols = symbols.argmax(dim=-1).cpu().numpy()  # [B, L]

            break  # single batch

    # Plot original vs reconstructed
    plt.figure(figsize=(10, 2 * num_examples))
    for i in range(num_examples):
        # Original
        plt.subplot(num_examples, 2, 2 * i + 1)
        plt.imshow(x[i, 0], cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # Reconstructed
        plt.subplot(num_examples, 2, 2 * i + 2)
        plt.imshow(x_recon[i, 0], cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

        # Print symbolic tokens
        b_tokens = ' '.join([f'b{tok}' for tok in symbols[i]])
        print(f"[Image {i + 1}] Bottleneck symbols:     {b_tokens}")
        print("")

    plt.tight_layout()
    plt.show()


def visualize_qna_prediction(answer_model, symbolic_model, dataset, idx=None, device='cpu'):
    """
    Visualizes the prediction of the model on a given index from the dataset.
    If idx is None, picks a random example.
    """
    import matplotlib.pyplot as plt
    import torch

    answer_model.eval()
    answer_model.to(device)
    symbolic_model.eval()
    symbolic_model.to(device)

    if idx is None:
        idx = torch.randint(0, len(dataset), (1,)).item()

    img, questions, answer, program, options = dataset[idx]
    img = img.unsqueeze(0).to(device)  # [1, 1, H, W]

    # Get reconstructed image and symbolic representation
    img_recon, symbols = symbolic_model(img, hard=True)

    img_np = img.squeeze().detach().cpu().numpy()
    img_recon_np = img_recon.squeeze().detach().cpu().numpy()

    questions = questions.unsqueeze(0).to(device)  # [1, 4, 1, H, W]

    with torch.no_grad():
        logits = answer_model(img_recon, questions)
        pred_idx = logits.argmax(dim=1).item()

    true_idx = answer.item()

    # Convert questions to numpy for plotting
    questions_np = questions.squeeze().detach().cpu().numpy()  # [4, H, W]

    # --- Plotting ---
    fig, axs = plt.subplots(1, 6, figsize=(15, 3))

    axs[0].imshow(img_np, cmap='gray')
    axs[0].set_title("Query Image")
    axs[0].axis('off')

    axs[1].imshow(img_recon_np, cmap='gray')
    axs[1].set_title("Reconstructed Image")
    axs[1].axis('off')

    print(f"[Image {1}] | Program: {program} | Questions: {options}")
    b_tokens = ' '.join([f'b{tok}' for tok in symbols])
    print(f"[Image {1}] Bottleneck symbols:     {b_tokens}")
    print("")

    for i in range(4):
        axs[i + 2].imshow(questions_np[i], cmap='gray')
        title = f"Q{i} "
        if i == true_idx:
            title += "✓"
        if i == pred_idx:
            title += " ⬅"
        axs[i + 2].set_title(title)
        axs[i + 2].axis('off')

    plt.suptitle(f"Predicted: Q{pred_idx} | Ground Truth: Q{true_idx}", fontsize=14)
    plt.tight_layout()
    plt.show()


def visualize_recon_from_msg(symbolic_model, msg, device="cuda"):
    symbolic_model.eval()

    symbols = {'A': 0, 'B': 1, 'C': 2, '0': 3, '1': 4, '2': 5, '+': 6, '*': 7}
    bneck_embedding = np.zeros((8, 8))

    locs = np.asarray([(i, symbols[c]) for i, c in enumerate(msg)])
    for loc in locs:
        bneck_embedding[loc[0], loc[1]] = 1

    bneck_embedding = np.expand_dims(bneck_embedding, axis=0)

    with torch.no_grad():
        bneck_embedding = torch.tensor(bneck_embedding.astype(np.float32)).to(device)
        x_recon = symbolic_model.recon_from_symbols(bneck_embedding)
        x_recon = x_recon.cpu().numpy()[0, 0, :, :]

    # Plot reconstructed
    plt.figure(figsize=(5, 5))

    # Reconstructed
    plt.imshow(x_recon, cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")
    plt.show()


def visualize_gumbel_reconstructions(sender, receiver, dataloader, device="cuda", num_examples=5):
    sender.eval()
    receiver.eval()

    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)  # [B, 1, 47, 41]
            symbols = sender(x, hard=True)  # [B, L, K]
            x_recon = receiver(symbols)

            # Convert symbolic tensor to human-readable format
            token_ids = symbols.argmax(dim=-1).cpu().numpy()  # [B, L]

            x = x.cpu().numpy()
            x_recon = x_recon.cpu().numpy()
            break  # One batch only

    # --- Plot ---
    plt.figure(figsize=(10, 2 * num_examples))
    for i in range(num_examples):
        # Original
        plt.subplot(num_examples, 2, 2 * i + 1)
        plt.imshow(x[i, 0], cmap='gray')
        plt.title("Original")
        plt.axis("off")

        # Reconstructed
        plt.subplot(num_examples, 2, 2 * i + 2)
        plt.imshow(x_recon[i, 0], cmap='gray')
        plt.title("Reconstructed")
        plt.axis("off")

        # Print symbolic sequence
        symbols_str = ' '.join([f'z{tok}' for tok in token_ids[i]])
        print(f"[Image {i + 1}] Symbolic language: {symbols_str}")

    plt.tight_layout()
    plt.show()