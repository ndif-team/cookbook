"""Standalone FactoredMatrix utility.

Extracted from TransformerLens (https://github.com/TransformerLensOrg/TransformerLens)
by Neel Nanda & Joseph Bloom. Licensed under MIT.

Represents a matrix as a product of two matrices (A @ B) and provides efficient
algorithms for SVD, eigenvalues, norms, and matrix operations without ever
materializing the full (potentially huge) product matrix.

Only dependency: PyTorch.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Tuple, Union, overload

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Tiny helpers (inlined from transformer_lens.utils so this file is standalone)
# ---------------------------------------------------------------------------

def _transpose(tensor: Tensor) -> Tensor:
    """Swap the last two dimensions of a tensor."""
    return tensor.transpose(-1, -2)


def _get_corner(tensor: Any, n: int = 3):
    """Return the top-left n×n corner of a tensor (useful for quick inspection)."""
    if isinstance(tensor, Tensor):
        return tensor[tuple(slice(n) for _ in range(tensor.ndim))]
    elif isinstance(tensor, FactoredMatrix):
        return tensor[tuple(slice(n) for _ in range(tensor.ndim))].AB


# ---------------------------------------------------------------------------
# FactoredMatrix
# ---------------------------------------------------------------------------

class FactoredMatrix:
    """Low-rank factored matrix M = A @ B.

    A is (..., ldim, mdim) and B is (..., mdim, rdim), so M is (..., ldim, rdim).
    The middle dimension *mdim* is typically much smaller than ldim or rdim,
    making it wasteful to materialise M. This class provides:

    - Efficient SVD (via the small mdim×mdim inner product)
    - Eigenvalues (via the smaller B @ A product)
    - Frobenius norm (from singular values)
    - Matrix multiplication that preserves the factored form
    - Indexing, scalar multiplication, transpose, etc.
    """

    def __init__(
        self,
        A: Tensor,  # (..., ldim, mdim)
        B: Tensor,  # (..., mdim, rdim)
    ):
        self.A = A
        self.B = B
        assert self.A.size(-1) == self.B.size(-2), (
            f"Factored matrix must match on inner dimension, "
            f"shapes were a: {self.A.shape}, b: {self.B.shape}"
        )
        self.ldim = self.A.size(-2)
        self.rdim = self.B.size(-1)
        self.mdim = self.B.size(-2)
        self.has_leading_dims = (self.A.ndim > 2) or (self.B.ndim > 2)
        self.shape = torch.broadcast_shapes(
            self.A.shape[:-2], self.B.shape[:-2]
        ) + (self.ldim, self.rdim)
        self.A = self.A.broadcast_to(self.shape[:-2] + (self.ldim, self.mdim))
        self.B = self.B.broadcast_to(self.shape[:-2] + (self.mdim, self.rdim))

    # ---- Matrix multiplication ----

    @overload
    def __matmul__(
        self,
        other: Union[Tensor, "FactoredMatrix"],
    ) -> "FactoredMatrix": ...

    @overload
    def __matmul__(self, other: Tensor) -> Tensor: ...  # type: ignore

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            if other.ndim < 2:
                return (self.A @ (self.B @ other.unsqueeze(-1))).squeeze(-1)
            else:
                assert other.size(-2) == self.rdim, (
                    f"Right matrix must match on inner dimension, "
                    f"shapes were self: {self.shape}, other: {other.shape}"
                )
                if self.rdim > self.mdim:
                    return FactoredMatrix(self.A, self.B @ other)
                else:
                    return FactoredMatrix(self.AB, other)
        elif isinstance(other, FactoredMatrix):
            return (self @ other.A) @ other.B

    def __rmatmul__(self, other):
        if isinstance(other, Tensor):
            assert other.size(-1) == self.ldim, (
                f"Left matrix must match on inner dimension, "
                f"shapes were self: {self.shape}, other: {other.shape}"
            )
            if other.ndim < 2:
                return ((other.unsqueeze(-2) @ self.A) @ self.B).squeeze(-2)
            elif self.ldim > self.mdim:
                return FactoredMatrix(other @ self.A, self.B)
            else:
                return FactoredMatrix(other, self.AB)
        elif isinstance(other, FactoredMatrix):
            return other.A @ (other.B @ self)

    # ---- Scalar multiplication ----

    def __mul__(self, scalar: Union[int, float, Tensor]) -> FactoredMatrix:
        if isinstance(scalar, Tensor):
            assert scalar.numel() == 1, (
                f"Tensor must be a scalar for use with * but was of shape {scalar.shape}. "
                f"For matrix multiplication, use @ instead."
            )
        return FactoredMatrix(self.A * scalar, self.B)

    def __rmul__(self, scalar: Union[int, float, Tensor]) -> FactoredMatrix:
        return self * scalar

    # ---- Products ----

    @property
    def AB(self) -> Tensor:
        """The full product matrix.  Expensive — may consume a lot of GPU memory."""
        return self.A @ self.B

    @property
    def BA(self) -> Tensor:
        """Reverse product B @ A.  Only valid when ldim == rdim."""
        assert self.rdim == self.ldim, (
            f"Can only compute BA when ldim == rdim, shape is {self.shape}"
        )
        return self.B @ self.A

    # ---- Transpose ----

    @property
    def T(self) -> FactoredMatrix:
        return FactoredMatrix(self.B.transpose(-2, -1), self.A.transpose(-2, -1))

    # ---- SVD (efficient via the small mdim×mdim core) ----

    @lru_cache(maxsize=None)
    def svd(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Efficient SVD → (U, S, Vh).

        U  @ diag(S) @ Vh.T == A @ B

        Complexity is dominated by SVD of the mdim×mdim inner matrix rather
        than the full ldim×rdim product.
        """
        Ua, Sa, Vha = torch.svd(self.A)
        Ub, Sb, Vhb = torch.svd(self.B)
        middle = Sa[..., :, None] * _transpose(Vha) @ Ub * Sb[..., None, :]
        Um, Sm, Vhm = torch.svd(middle)
        U = Ua @ Um
        Vh = Vhb @ Vhm
        S = Sm
        return U, S, Vh

    @property
    def U(self) -> Tensor:
        return self.svd()[0]

    @property
    def S(self) -> Tensor:
        return self.svd()[1]

    @property
    def Vh(self) -> Tensor:
        return self.svd()[2]

    # ---- Eigenvalues ----

    @property
    def eigenvalues(self) -> Tensor:
        """Eigenvalues of AB (== eigenvalues of BA, up to trailing zeros)."""
        mat = self.BA
        if mat.dtype in [torch.bfloat16, torch.float16]:
            mat = mat.to(torch.float32)
        return torch.linalg.eig(mat).eigenvalues

    # ---- Norm ----

    def norm(self) -> Tensor:
        """Frobenius norm = sqrt(sum of squared singular values)."""
        return self.S.pow(2).sum(-1).sqrt()

    # ---- Factorisation helpers ----

    def make_even(self) -> FactoredMatrix:
        """Re-factorise so each half carries sqrt of each singular value."""
        return FactoredMatrix(
            self.U * self.S.sqrt()[..., None, :],
            self.S.sqrt()[..., :, None] * _transpose(self.Vh),
        )

    def collapse_l(self) -> Tensor:
        """Remove left orthogonal factor → (…, mdim, rdim) tensor."""
        return self.S[..., :, None] * _transpose(self.Vh)

    def collapse_r(self) -> Tensor:
        """Remove right orthogonal factor → (…, ldim, mdim) tensor."""
        return self.U * self.S[..., None, :]

    # ---- Indexing ----

    def _convert_to_slice(self, sequence, idx):
        if isinstance(idx, int):
            sequence = list(sequence)
            if isinstance(sequence[idx], int):
                sequence[idx] = slice(sequence[idx], sequence[idx] + 1)
            sequence = tuple(sequence)
        return sequence

    def __getitem__(self, idx) -> FactoredMatrix:
        """Index into leading dimensions (and optionally the matrix dims)."""
        if not isinstance(idx, tuple):
            idx = (idx,)
        length = len([i for i in idx if i is not None])
        if length <= len(self.shape) - 2:
            return FactoredMatrix(self.A[idx], self.B[idx])
        elif length == len(self.shape) - 1:
            idx = self._convert_to_slice(idx, -1)
            return FactoredMatrix(self.A[idx], self.B[idx[:-1]])
        elif length == len(self.shape):
            idx = self._convert_to_slice(idx, -1)
            idx = self._convert_to_slice(idx, -2)
            return FactoredMatrix(
                self.A[idx[:-1]],
                self.B[idx[:-2] + (slice(None), idx[-1])],
            )
        else:
            raise ValueError(
                f"{idx} is too long an index for a FactoredMatrix with shape {self.shape}"
            )

    # ---- Misc ----

    def get_corner(self, k: int = 3):
        """Quick peek at the top-left k×k corner of the full product."""
        return _get_corner(self.A[..., :k, :] @ self.B[..., :, :k], k)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def unsqueeze(self, k: int) -> FactoredMatrix:
        return FactoredMatrix(self.A.unsqueeze(k), self.B.unsqueeze(k))

    @property
    def pair(self) -> Tuple[Tensor, Tensor]:
        """Return the raw (A, B) factor pair."""
        return (self.A, self.B)

    def __repr__(self):
        return f"FactoredMatrix: Shape({self.shape}), Hidden Dim({self.mdim})"
