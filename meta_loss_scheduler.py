"""
SpectraMind V50 – Meta Loss Scheduler
-------------------------------------
Dynamically adjusts weighting of symbolic vs. supervised loss components
based on epoch, violation frequency, or curriculum stage.
Supports ramping, cyclical, stepwise, and curriculum-sensitive modes.
"""

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
            schedule_type: 'linear_ramp', 'constant', 'cyclical', 'step', or 'curriculum'
            max_weight: maximum symbolic loss scaling
            warmup_epochs: number of epochs to ramp up
            total_epochs: total training epochs (used for cyclical or curriculum)
            step_interval: interval for step increase
            curriculum_epochs: optional list of (epoch, weight) tuples
        """
        self.schedule_type = schedule_type
        self.max_weight = max_weight
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.step_interval = step_interval
        self.curriculum_epochs = curriculum_epochs or []

    def get_weight(self, epoch: int) -> float:
        if self.schedule_type == "constant":
            return self.max_weight

        elif self.schedule_type == "linear_ramp":
            if epoch < self.warmup_epochs:
                return (epoch / self.warmup_epochs) * self.max_weight
            return self.max_weight

        elif self.schedule_type == "cyclical":
            cycle = (epoch % self.total_epochs) / self.total_epochs
            return 0.5 * self.max_weight * (1 + abs(2 * cycle - 1))

        elif self.schedule_type == "step":
            steps = epoch // self.step_interval
            weight = min(self.max_weight, (steps + 1) * (self.max_weight / (self.total_epochs // self.step_interval)))
            return weight

        elif self.schedule_type == "curriculum":
            applicable = [w for e, w in self.curriculum_epochs if epoch >= e]
            return applicable[-1] if applicable else 0.0

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

if __name__ == "__main__":
    print("Testing MetaLossScheduler variants")

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
