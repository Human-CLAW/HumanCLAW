"""Optional motion-data and training tools.

Nothing in this package is imported by rollout.  The separation keeps the
evaluation install small while letting training reuse the exact network
definitions used by the released motion runtime.
"""

__all__: list[str] = []
