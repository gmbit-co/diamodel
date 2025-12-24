from dataclasses import dataclass, asdict

__all__ = ["Config"]


@dataclass
class Config:
    """Configuration for diabetes model training and prediction."""

    name: str  # config name
    dt: int  # cgm readings interval, mins
    maxact: int  # maximum carbs/insulin activity duration, ticks
    maxpred: int  # prediction horizon, ticks
    gscale: float  # translates ticks to gamma dist: x = gscale * tick
    nsamples: int  # num samples to keep for inference

    @staticmethod
    def default_config(name: str = "default", dt=5) -> "Config":
        """Create a Config instance with default parameters."""
        # all times are in mins
        maxact = round(9 * 60 / dt)
        maxpred = round(4 * 60 / dt)

        # 1h = 2 gamma, 1 tick = `gscale` gamma
        gscale = 2.0 / (60 / dt)
        nsamples = 500

        return Config(
            name=name,
            dt=dt,
            maxact=maxact,
            maxpred=maxpred,
            gscale=gscale,
            nsamples=nsamples,
        )

    def asdict(self):
        return asdict(self)
