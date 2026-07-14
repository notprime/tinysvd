import torch

from . import utils
from . import eigendecomposition

class SVD():
    def __init__(
            self,
            tolerance: float | None = 1e-4,
            max_iterations: int = 1000,
            seed: int | None = 23,
            ):
        
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.seed = seed
        
        # Initialize a random generator
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

        self.U = None
        self.Sigma = None
        self.V = None
        self.singular_values = None

    def _compute_ata(self, A):
        ata = utils.compute_ATA(A)
        # aggiungere assert per check su dimensioni, per sicurezza
        return ata
    
    def _compute_left_singular_vectors(self, A, V, singular_values):
        return utils.compute_left_singular_vectors(
            A,
            V,
            singular_values
        )
    
    def _build_sigma(self, singular_values, rows, cols):
        # ThinSVD, Sigma will be a square matrix
        return utils.build_sigma(
            singular_values,
            #rows,
            #cols
        )
    
    def reconstruct(self):
        return self.U @ self.Sigma @ self.V.T
    
    def fit(self, A):
        ATA = self._compute_ata(A)

        eigenvalues, V = eigendecomposition.eigendecompose(
            ATA,
            generator=self.rng,
            tolerance=self.tolerance,
            )
        singular_values = torch.sqrt(eigenvalues)

        U = self._compute_left_singular_vectors(
            A,
            V,
            singular_values
        )

        Sigma = self._build_sigma(
            singular_values,
            U.shape[1],
            V.shape[1]
        )

        # aggiungere checks vari: shapes di V, Sigma, V;
        # U.T @ U circa I
        # V.T @ V circa I
        # finally A circa U # Sigma @ V.T
        self.U = U
        self.Sigma = Sigma
        self.V = V
        self.singular_values = singular_values

        return self
