import torch


def normalize(vector: torch.Tensor) -> torch.Tensor:
    """
    Normalize a vector to unit Euclidean norm

    Args:
        vector: torch.Tensor, input vector

    Returns:
        torch.Tensor, unit-norm version of the input vector
    """

    norm = torch.linalg.norm(vector)

    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")

    return vector / norm


def compute_ATA(A: torch.Tensor) -> torch.Tensor:
    """
    Compute A.T@A
    
    Args:
        A: torch.Tensor, input matrix
        
    Returns:
        torch.Tensor, symmetric positive semi-definite matrix A.T@A
    """

    return A.T @ A


def rayleigh_quotient(matrix: torch.Tensor,
                      vector: torch.Tensor) -> torch.Tensor:
    """
    Compute the Rayleigh quotiens. The vector is assumed to be normalized, such that:
    
    lambda = v.T @ M @ v

    Args:
        matrix: torch.Tensor, input matrix
        vector: torch.Tensor, input vector

    Returns:
        torch.Tensor, dominant eigenvalue estimate
    """
    # vector is 1D, .T is not needed, torch only transpose if dim>=2
    return vector @ matrix @ vector


def build_sigma(singular_vals: torch.Tensor) -> torch.Tensor:
    """
    Build the rectangular Sigma matrix
    
    Args:
        singular_vals: torch.Tensor, singular values sorted in descending order
        
    Returns:
        torch.Tensor, square Sigma matrix"""
    
    return torch.diag(singular_vals)


def compute_left_singular_vectors(
        A: torch.Tensor,
        V: torch.Tensor,
        singular_values: torch.Tensor,
) -> torch.Tensor:
    
    U_cols = []
    for sigma, v in zip(singular_values, V.T):
        u = A @ v
        u = u / sigma
        U_cols.append(u)

    return torch.stack(U_cols, dim=1)


def reconstruct_matrix(U, Sigma, V):
    """todo properly"""
    return U @ Sigma @ V.T

