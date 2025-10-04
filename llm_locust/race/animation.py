"""
Animation utilities for smooth visual transitions.

Provides interpolation and easing functions for polished UI animations.
"""

import time


class AnimatedValue:
    """
    Smoothly animates a numeric value over time.

    Uses linear interpolation to avoid sudden jumps in displayed numbers.
    """

    def __init__(self, initial_value: float = 0, speed: float = 2.0) -> None:
        """
        Initialize animated value.

        Args:
            initial_value: Starting value
            speed: Animation speed (units per second)
        """
        self.current = initial_value
        self.target = initial_value
        self.speed = speed
        self.last_update = time.time()

    def set_target(self, target: float) -> None:
        """Set new target value to animate towards."""
        self.target = target

    def update(self) -> float:
        """
        Update current value by interpolating towards target.

        Returns:
            Current interpolated value
        """
        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        if abs(self.current - self.target) < 0.1:
            # Close enough, snap to target
            self.current = self.target
        else:
            # Interpolate towards target
            diff = self.target - self.current
            step = diff * min(dt * self.speed, 1.0)
            self.current += step

        return self.current

    @property
    def value(self) -> float:
        """Get current value (updates automatically)."""
        return self.update()


class CounterAnimation:
    """Animated counter that counts up/down smoothly."""

    def __init__(self, initial: int = 0, speed: float = 10.0) -> None:
        """
        Initialize counter animation.

        Args:
            initial: Starting value
            speed: Count speed (units per second)
        """
        self.animated = AnimatedValue(float(initial), speed)

    def set(self, target: int) -> None:
        """Set target count."""
        self.animated.set_target(float(target))

    def get(self) -> int:
        """Get current animated count (rounded)."""
        return int(self.animated.value)


def ease_in_out(t: float) -> float:
    """
    Ease-in-out interpolation function.

    Args:
        t: Time value (0.0 to 1.0)

    Returns:
        Eased value (0.0 to 1.0)
    """
    if t < 0.5:
        return 2 * t * t
    else:
        return 1 - 2 * (1 - t) * (1 - t)


def lerp(start: float, end: float, t: float) -> float:
    """
    Linear interpolation between two values.

    Args:
        start: Start value
        end: End value
        t: Interpolation factor (0.0 to 1.0)

    Returns:
        Interpolated value
    """
    return start + (end - start) * t


class ProgressAnimation:
    """Smoothly animate progress bar advancement."""

    def __init__(self, total: int = 100) -> None:
        """
        Initialize progress animation.

        Args:
            total: Total progress (100%)
        """
        self.total = total
        self.current = AnimatedValue(0, speed=5.0)
        self.target_value = 0

    def set_progress(self, current: int) -> None:
        """Set target progress."""
        self.target_value = current
        self.current.set_target(float(current))

    def get_progress(self) -> int:
        """Get current animated progress."""
        return int(self.current.value)

    def get_percentage(self) -> float:
        """Get current percentage (0-100)."""
        if self.total == 0:
            return 0.0
        return (self.current.value / self.total) * 100

