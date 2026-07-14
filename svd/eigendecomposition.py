import torch

from . import utils


def power_iteration(matrix: torch.Tensor,
                    generator: torch.Generator,
                    max_iterations: int = 1000,
                    tolerance: float = 1e-8,
                    seed: int | None = None,) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the dominant eigenpair of a symmetric matrix using
    the Power Iteration algorithm.
    
    Args:
        matrix: torch.Tensor, symmetric input matrix
        max_iterations: int, maximum number of iterations
        tolerance: float, convergence threshold
        seed: int | None, random seed used to initialize the starting vector
        
    Returns:
        eigenvalue: torch.Tensor, dominant eigenvalue
        v: torch.Tensor, corresponding normalized eigenvector"""

    # Initialize random unit vector and normalize it
    v = torch.randn(
        matrix.shape[1],
        dtype=matrix.dtype,
        device=matrix.device,
        generator=generator
    )
    v = utils.normalize(v)

    # Compute the dominant eigenvector
    for _ in range(max_iterations): # safety guard
        w = matrix @ v
        v = utils.normalize(w)

        # Check the residual norm: r = Mv - (lambda)v
        lambda_est = utils.rayleigh_quotient(matrix, v)
        residual = matrix @ v - lambda_est*v
        if torch.linalg.norm(residual) < tolerance:
            eigenvalue = lambda_est
            break
    
    else:
        raise RuntimeError(
            "Power Iteration failed to converge."
        )

    # Compute the eigenvalue
    #eigenvalue = utils.rayleigh_quotient(matrix, v)
    
    return eigenvalue, v


def deflate(matrix: torch.Tensor,
            eigenvalue: torch.Tensor,
            eigenvector: torch.Tensor,) -> torch.Tensor:
    """
    Remove an eigenpair from a symmetric matrix using
    rank-1 deflation.
    
    Args:
        matrix: torch.Tensor, symmetric matrix
        eigenvalue: torch.Tensor, dominant eigenvalue
        eigenvector: torch.Tensor, corresponding normalized eigenvector
        
    Returns:
        deflated: torch.Tensor, deflated matrix
    """

    deflated = matrix - eigenvalue * torch.outer(eigenvector, eigenvector)

    return deflated


def eigendecompose(
        matrix: torch.Tensor,
        generator: torch.Generator,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the eigendecomposition of a symmetric matrix
    using Power Iteration and Deflation.

    Args:
        matrix : torch.Tensor
        max_iterations : int
        tolerance : float

    Returns:
        eigenvalues : torch.Tensor
        eigenvectors : torch.Tensor
    """
    
    # Clone the matrix
    current_matrix = matrix.clone()

    eigenvalues = []
    eigenvectors = []

    # n eigenpairs in a nxn matrix
    n = matrix.shape[0]

    while torch.linalg.norm(current_matrix) > tolerance:
        eigenvalue, eigenvector = power_iteration(
            current_matrix,
            generator=generator,
            max_iterations=max_iterations,
            tolerance=tolerance)
        
        if eigenvalue < tolerance:
            # extra safeguard, eigenvalue close to zero
            break

        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)

        # Deflate
        current_matrix = deflate(
            current_matrix,
            eigenvalue,
            eigenvector,)

        # Numerical correction
        current_matrix = 0.5 * (current_matrix + current_matrix.T)

    eigenvalues = torch.stack(eigenvalues)
    eigenvectors = torch.stack(eigenvectors, dim=1) # V = [v1 v2 ...]

    # ort eigenvalues for safety (should already be sorted)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors