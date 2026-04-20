"""Early stopping."""

from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int, *, maximize: bool = True) -> None:
        self.patience = patience
        self.maximize = maximize
        self.best_score: float | None = None
        self.num_bad_epochs = 0

    def step(self, score: float | None) -> bool:
        if score is None:
            self.num_bad_epochs += 1
            return self.num_bad_epochs > self.patience
        if self.best_score is None:
            self.best_score = score
            self.num_bad_epochs = 0
            return False
        improved = score > self.best_score if self.maximize else score < self.best_score
        if improved:
            self.best_score = score
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs > self.patience
