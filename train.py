# train.py
import torch
from model import CLM8B


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing CLM-8B on {device}...")

    # Load architecture
    model = CLM8B(base_model_name="Qwen/Qwen2.5-7B", load_in_4bit=True)
    model.state_head.to(device)
    model.action_head.to(device)
    model.logit_scale.data = model.logit_scale.data.to(device)

    # Trainable parameters are ONLY the heads and scale (~18M params)
    optimizer = torch.optim.AdamW(
        list(model.state_head.parameters())
        + list(model.action_head.parameters())
        + [model.logit_scale],
        lr=3e-4,
        weight_decay=0.01,
    )

    # Sample dataset (can be substituted with Nemotron-CC or custom JSONL)
    train_pairs = [
        ("Cactus obstacle detected 60px ahead", "JUMP"),
        ("Low flying pterodactyl overhead", "DUCK"),
        ("Clear ground terrain ahead", "RUN"),
        ("Customer requesting refund on duplicate invoice", "ROUTE_BILLING"),
        ("Server API latency spiking with 504 errors", "ROUTE_DEVOPS"),
        ("User wants to schedule a sync at 4 PM", "CALL_CALENDAR"),
        ("User asks for current stock price of Apple", "CALL_FINANCE"),
    ]

    states = [p[0] for p in train_pairs]
    actions = [p[1] for p in train_pairs]

    print("Beginning Training...")
    model.train()
    for step in range(1, 41):
        loss = model(states, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0 or step == 1:
            print(
                f"Step {step:02d} | Contrastive InfoNCE Loss: {loss.item():.4f} | Scale: {model.logit_scale.exp().item():.2f}"
            )

    # Save ONLY the 75MB weights
    checkpoint = {
        "state_head": model.state_head.state_dict(),
        "action_head": model.action_head.state_dict(),
        "logit_scale": model.logit_scale.data,
    }
    torch.save(checkpoint, "clm_8b_heads.pt")
    print("\nSaved trained weights to clm_8b_heads.pt (~75MB).")


if __name__ == "__main__":
    run_training()
