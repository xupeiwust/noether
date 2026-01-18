#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pydantic import BaseModel, Field


class ContinuousSincosEmbeddingConfig(BaseModel):
    """Configuration for Continuous Sine-Cosine Embedding layer."""

    hidden_dim: int = Field(...)
    """Dimensionality of the output embedding."""
    input_dim: int = Field(...)
    """Dimensionality of the input coordinates."""
    max_wavelength: int = Field(10000)
    """Maximum wavelength for the sine-cosine embeddings."""
