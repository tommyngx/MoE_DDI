import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from plotting import plot_training_history

def generate_demo_data(num_epochs=50):
    history = []
    base_lr_1st = 5e-5
    base_lr_2nd = 1e-5
    t_max = 50
    
    for ep in range(1, num_epochs + 1):
        # Cosine annealing factor
        cosine_factor = 0.5 * (1 + math.cos(math.pi * ep / t_max))
        lr_1st = base_lr_1st * cosine_factor
        lr_2nd = base_lr_2nd * cosine_factor

        # Realistic metrics simulation
        progress = ep / num_epochs
        train_loss = 2.5 * math.exp(-3 * progress) + 0.3 + 0.02 * math.sin(ep)
        val_loss = 2.6 * math.exp(-2.5 * progress) + 0.35 + 0.03 * math.cos(ep)
        accuracy = 0.2 + 0.65 * (1 - math.exp(-4 * progress)) + 0.01 * math.sin(ep)
        macro_f1 = 0.1 + 0.60 * (1 - math.exp(-3.5 * progress)) + 0.01 * math.cos(ep)
        
        balance_loss = 0.05 * math.exp(-progress) + 0.005
        router_z_loss = 0.01 * math.exp(-0.5 * progress) + 0.001
        moe_aux = 0.4 * math.exp(-2 * progress) + 0.08
        global_aux = 0.35 * math.exp(-2.2 * progress) + 0.07

        history.append({
            "epoch": ep,
            "learning_rate": lr_1st,
            "learning_rate_1st": lr_1st,
            "learning_rate_2nd": lr_2nd,
            "train_classification_loss": train_loss,
            "train_balance_loss": balance_loss,
            "train_router_z_loss": router_z_loss,
            "train_moe_auxiliary_loss": moe_aux,
            "train_global_auxiliary_loss": global_aux,
            "elapsed_seconds": ep * 15.0,
            "validation": {
                "loss": val_loss,
                "accuracy": accuracy,
                "macro_f1": macro_f1
            }
        })
    return history

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "demo_output"
    out_dir.mkdir(exist_ok=True)
    history = generate_demo_data(50)
    plot_training_history(history, out_dir, epoch=50, dataset_tag="demo")
    print(f"Plot saved to: {out_dir / 'training_demo.png'}")
