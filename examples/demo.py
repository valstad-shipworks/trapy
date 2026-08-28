"""Example usage of trapy.

Install the package with maturin from the `trapy/` directory:

    cd trapy
    maturin develop --release        # editable install into the active venv
    python examples/demo.py

A tracing subscriber is not configured here, so events are dispatched but not
printed — wire one up on the Rust side (e.g. `tracing_subscriber::fmt`) to see
the structured output.
"""

from dataclasses import dataclass

import trapy


@dataclass
class Pose:
    x: float
    y: float
    label: str


@dataclass
class Path:
    name: str
    poses: list[Pose]


def main() -> None:
    # Plain message — no kwargs.
    trapy.info("starting demo")

    # Primitives as kwargs.
    trapy.debug(
        "config loaded",
        retries=3,
        timeout=2.5,
        verbose=True,
        env="staging",
    )

    # Collections — list/tuple/dict with arbitrary nesting.
    trapy.info(
        "batch summary",
        sizes=[1, 2, 3, 5, 8],
        bounds=(0.0, 1.0),
        meta={"shard": "a", "tags": ["fast", "retry"], "stats": {"ok": 12, "err": 0}},
    )

    # Dataclass — walked via dataclasses.fields().
    home = Pose(x=0.0, y=0.0, label="home")
    trapy.info("at home pose", pose=home)

    # Nested dataclass inside a list inside a dataclass.
    path = Path(
        name="weld_seam_3",
        poses=[Pose(0.0, 0.0, "a"), Pose(1.0, 0.5, "b"), Pose(2.0, 0.0, "c")],
    )
    trapy.warn("path candidate", path=path, score=0.87)

    # Errors.
    try:
        raise RuntimeError("simulated failure")
    except RuntimeError as exc:
        trapy.error("operation failed", reason=str(exc), code=500)

    # numpy arrays / scalars (and anything else exposing `__array__`).
    try:
        import numpy as np

        trapy.info(
            "numpy values",
            scalar=np.float64(1.5),
            row=np.array([1.0, 2.0, 3.0]),
            grid=np.arange(6).reshape(2, 3),
            flag=np.bool_(True),
        )
    except ImportError:
        pass

    # Trace — most verbose.
    for i in range(3):
        trapy.trace("loop tick", i=i, ratio=i / 3.0)

    # No-message form is allowed; kwargs still flow through.
    trapy.info(x=1, y=2)

    print("demo complete")


if __name__ == "__main__":
    main()
