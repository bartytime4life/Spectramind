"""
SpectraMind V50 – Meta Loss Scheduler
-------------------------------------
Dynamic symbolic loss weighting controller supporting multiple scheduling strategies:
- constant
- linear ramp
- cyclical triangle
- stepwise ramp
- curriculum-based epoch mapping

Logs, plots, and previews symbolic weights for debug and diagnostics.
"""

import matplotlib.pyplot as plt

class MetaLossScheduler:
    def __init__(
        self,
        schedule_type="linear_ramp",
        max_weight=1.0,
        warmup_epochs=5,
        total_epochs=50,
        step_interval=10,
        curriculum_epochs=None
    ):
        """
        Args:
            schedule_type: one of ['linear_ramp', 'constant', 'cyclical', 'step', 'curriculum']
            max_weight: maximum symbolic loss multiplier
            warmup_epochs: ramping period for linear ramp
            total_epochs: number of total training epochs (used by cyclical/step)
            step_interval: interval size for step-wise schedule
            curriculum_epochs: list of (epoch, weight) overrides
        """
        self.schedule_type = schedule_type
        self.max_weight = max_weight
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.step_interval = step_interval
        self.curriculum_epochs = sorted(curriculum_epochs or [])

    def get_weight(self, epoch: int) -> float:
        if self.schedule_type == "constant":
            return self.max_weight

        elif self.schedule_type == "linear_ramp":
            if epoch < self.warmup_epochs:
                return (epoch / self.warmup_epochs) * self.max_weight
            return self.max_weight

        elif self.schedule_type == "cyclical":
            phase = (epoch % self.total_epochs) / self.total_epochs
            return 0.5 * self.max_weight * (1 + abs(2 * phase - 1))

        elif self.schedule_type == "step":
            step_num = epoch // self.step_interval
            max_steps = self.total_epochs // self.step_interval
            weight = (step_num + 1) * (self.max_weight / max_steps)
            return min(weight, self.max_weight)

        elif self.schedule_type == "curriculum":
            for e, w in reversed(self.curriculum_epochs):
                if epoch >= e:
                    return w
            return 0.0

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def plot(self, max_epochs=50, save_path="outputs/scheduler_preview.png"):
        weights = [self.get_weight(e) for e in range(max_epochs)]
        plt.figure(figsize=(8, 3))
        plt.plot(weights, color="blue", linewidth=2)
        plt.title(f"Symbolic Weight Schedule – {self.schedule_type}")
        plt.xlabel("Epoch")
        plt.ylabel("Symbolic Loss Weight")
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"📈 Schedule preview saved to {save_path}")

if __name__ == "__main__":
    schedulers = {
        "linear_ramp": MetaLossScheduler("linear_ramp", max_weight=1.0, warmup_epochs=5),
        "constant": MetaLossScheduler("constant", max_weight=0.8),
        "cyclical": MetaLossScheduler("cyclical", max_weight=1.0, total_epochs=10),
        "step": MetaLossScheduler("step", max_weight=1.0, total_epochs=20, step_interval=5),
        "curriculum": MetaLossScheduler("curriculum", max_weight=1.0, curriculum_epochs=[(0, 0.0), (5, 0.3), (10, 0.6), (15, 1.0)])
    }

    for name, sched in schedulers.items():
        print(f"\n{name.upper()}:")
        for epoch in range(20):
            w = sched.get_weight(epoch)
            print(f"Epoch {epoch:2d}: Weight = {w:.3f}")
        sched.plot(max_epochs=20, save_path=f"outputs/schedule_{name}.png")
