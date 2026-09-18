##########################################################################
#                                 SFT training 
##########################################################################

import torch


@torch.no_grad()
def estimate_loss(
    model,
    train_loader,
    val_loader,
    eval_iters,
    device
):

    result = {}

    model.eval()

    for split, loader in [
        ("train", train_loader),
        ("val", val_loader)
    ]:

        losses = []

        for i, (x, y, loss_mask) in enumerate(loader):

            if i >= eval_iters:
                break

            x = x.to(device)
            y = y.to(device)
            loss_mask = loss_mask.to(device)

            _, loss = model(
                x,
                y,
                loss_mask
            )

            losses.append(loss.item())

        result[split] = sum(losses) / len(losses)

    model.train()

    return result


def train(
    model,
    train_loader,
    val_loader,
    eval_interval,
    eval_iters,
    n_epochs,
    learning_rate,
    device,
    patience=4,
    min_delta=0.01,
    best_model_path="best_sft_model.pt"
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    model.to(device)
    model.train()

    # ------------------------------
    # Early stopping variables
    # ------------------------------

    best_val_loss = float("inf")
    evaluations_without_improvement = 0

    step = 0
    stop_training = False

    # ------------------------------
    # Training
    # ------------------------------

    for epoch in range(n_epochs):

        for batch_idx, (x, y, loss_mask) in enumerate(train_loader):

            x = x.to(device)
            y = y.to(device)
            loss_mask = loss_mask.to(device)

            _, loss = model(
                x,
                y,
                loss_mask
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # ------------------------------
            # Evaluation
            # ------------------------------

            if step % eval_interval == 0:

                losses = estimate_loss(
                    model,
                    train_loader,
                    val_loader,
                    eval_iters,
                    device
                )

                train_loss = losses["train"]
                val_loss = losses["val"]

                print(
                    f"step {step}: "
                    f"train loss {train_loss:.4f}, "
                    f"val loss {val_loss:.4f}"
                )

                # ------------------------------
                # Improvement?
                # ------------------------------

                if val_loss < best_val_loss - min_delta:

                    best_val_loss = val_loss

                    evaluations_without_improvement = 0

                    torch.save(
                        model.state_dict(),
                        best_model_path
                    )

                    print(
                        f"  new best validation loss: "
                        f"{best_val_loss:.4f}"
                    )

                else:

                    evaluations_without_improvement += 1

                    print(
                        "  no improvement: "
                        f"{evaluations_without_improvement}/"
                        f"{patience}"
                    )

                # ------------------------------
                # Early stopping
                # ------------------------------

                if evaluations_without_improvement >= patience:

                    print(
                        "\nEarly stopping triggered."
                    )

                    stop_training = True
                    break

            step += 1

        if stop_training:
            break

    # ====================================================
    # Restore BEST weights
    # ====================================================

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device,
            weights_only=True
        )
    )

    print(
        f"\nRestored best model "
        f"(val loss = {best_val_loss:.4f})"
    )

    return model