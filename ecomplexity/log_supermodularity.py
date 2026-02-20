"""
Log-supermodularity is defined in detail here: https://growthlab.hks.harvard.edu/publications/structural-ranking-economic-complexity

Brief description:
Log-supermodularity is a property that is a sufficient condition for the complexity values (ECI and PCI) to be a true representation of the sophistication or complexity of the location and of the products. If log-supermodularity is not satisfied, then those values could still be representative of the complexity, but it is not guaranteed.

The log-supermodularity of the matrix is calculated as follows:

The matrix $M$ is log-supermodular if the following condition is satisfied:

For every pair of countries $i^{'}$ and $i$ such that ${ECI}_{i^{'}} > {ECI}_i$, and for every pair of products $j^{'}$ and $j$ such that ${PCI}_{j^{'}} > {PCI}_j$, the following inequality holds:

$$\frac{M_{i^{'}j^{'}}}{M_{i^{'}j}} > \frac{M_{ij^{'}}}{M_{ij}}$$

In reality, this inequality doesn't hold for every pair of countries and products, but the percentage of pairs for which this inequality holds is calculated. If this percentage is high, then the matrix is considered to be sufficiently log-supermodular.
"""

import numpy as np
import warnings


def get_frac_logsupermodular(matrix, eci, pci, samples_to_use=None):
    """
    Check the log-supermodularity of a matrix.

    This function calculates the percentage of pairs of countries and products that
    satisfy the log-supermodularity condition. The log-supermodularity condition is
    checked for pairs of countries (i', i) where ECI[i'] > ECI[i] and pairs of
    products (j', j) where PCI[j'] > PCI[j].

    Args:
        matrix (numpy.ndarray): The input matrix (RCA, RPOP, or MCP).
        eci (numpy.ndarray): The ECI values for each country.
        pci (numpy.ndarray): The PCI values for each product.
        samples_to_use (optional): The sampling parameter that
            determines the number of countries and products to be used for the
            log-supermodularity check. Takes None, int, or "all".
            If None (default), the full set of countries and products is used,
            provided the number of elements of the matrix is less than 10,000.
            If the number of elements exceeds 10,000, then roughly 10,000 samples are used.
            If an integer value is provided, roughly that many samples will be used.
            If "all", all elements are used without sampling.

    Returns:
        float: The percentage of pairs that satisfy the log-supermodularity condition.
    """
    n_countries, n_products = matrix.shape
    num_elements = n_countries * n_products

    # Determine the sampling parameter based on the number of comparisons
    if isinstance(samples_to_use, bool):
        raise ValueError("samples_to_use cannot be a boolean value.")

    if samples_to_use is None:
        if num_elements >= 1e4:
            samples_to_use = 1e4
        else:
            samples_to_use = num_elements
    elif samples_to_use == "all":
        samples_to_use = num_elements
    elif isinstance(samples_to_use, int):
        pass
    else:
        raise ValueError(
            "samples_to_use should be None, an integer, or 'all'."
            f" Got {samples_to_use} of type {type(samples_to_use)}."
        )

    if samples_to_use > 1e4:
        warnings.warn(
            f"The number of samples used to compute log-supermodularity ({samples_to_use}), exceeds 10,000. May take a long time."
        )

    # Sample countries and products based on the sampling parameter
    if samples_to_use < num_elements:
        # Roughly use samples_to_use
        # Countries to use = total countries * samples_to_use / total elements
        # Products to use = total products * samples_to_use / total elements
        n_sampled_countries = int(np.floor(n_countries * samples_to_use / num_elements))
        n_sampled_products = int(np.floor(n_products * samples_to_use / num_elements))

        if n_sampled_products == 0:
            n_sampled_products = 1
        if n_sampled_countries == 0:
            n_sampled_countries = 1

        country_indices = np.random.choice(
            n_countries, size=n_sampled_countries, replace=False
        )
        product_indices = np.random.choice(
            n_products, size=n_sampled_products, replace=False
        )
    else:
        country_indices = np.arange(n_countries)
        product_indices = np.arange(n_products)

    # Filter matrix, eci and pci based on the sampled indices
    matrix = matrix[country_indices][:, product_indices]
    eci = eci[country_indices]
    pci = pci[product_indices]

    # Sort matrix based on ECI for rows and PCI for columns
    eci_order = np.argsort(eci)[::-1]
    pci_order = np.argsort(pci)[::-1]
    matrix = matrix[eci_order][:, pci_order]
    eci = eci[eci_order]
    pci = pci[pci_order]

    # Create meshgrids of country and product indices using np.ogrid
    country_idx, product_idx = np.ogrid[: matrix.shape[0], : matrix.shape[1]]
    country_idx_comparison = country_idx[:, :, np.newaxis, np.newaxis]
    product_idx_comparison = product_idx[:, :, np.newaxis, np.newaxis]

    # Since the matrix is sorted, any pair of elements i > i' and j > j' is a valid pair
    valid_pairs = (country_idx_comparison > country_idx) & (
        product_idx_comparison > product_idx
    )

    # Check the log-supermodularity condition
    # Broadcast matrix to four dimensions for different comparisons
    mijp = matrix.reshape(
        (1, matrix.shape[1], matrix.shape[0], 1)
    )  # M[i, j'], Shape (1, 4, 2, 1)
    mipj = matrix.reshape(
        (matrix.shape[0], 1, 1, matrix.shape[1])
    )  # M[i', j], Shape (2, 1, 1, 4)
    mipjp = matrix[:, :, np.newaxis, np.newaxis]  # M[i', j'], Shape (2, 4, 1, 1)
    mij = matrix[np.newaxis, np.newaxis, :, :]  # M[i, j], Shape (1, 1, 2, 4)

    # Calculate the ratios using valid_pairs to mask the operations
    with np.errstate(divide="ignore", invalid="ignore"):
        left_ratio = np.nan_to_num(
            np.where(valid_pairs, mipjp / mipj, np.nan), nan=0, posinf=0, neginf=0
        )
        right_ratio = np.nan_to_num(
            np.where(valid_pairs, mijp / mij, np.nan), nan=0, posinf=0, neginf=0
        )
    condition_met = left_ratio > right_ratio

    # Calculate percentage of conditions met
    total_valid = valid_pairs.sum()
    # The total number of valid pairs for a nxk matrix should be n*(n-1)*k*(k-1)/4
    n = matrix.shape[0]
    k = matrix.shape[1]
    expected_valid = n * (n - 1) * k * (k - 1) / 4
    if total_valid != expected_valid:
        warnings.warn(
            f"Expected {expected_valid} valid pairs, but found {total_valid}."
        )
    # If no valid pairs, then warn user
    if total_valid == 0:
        warnings.warn(
            "No valid pairs found for log-supermodularity check. "
            "This may indicate that the matrix is too small or the ECI and PCI values are not well-defined."
        )
        return 0.0
    fraction_log_supermodular = condition_met.sum() / total_valid

    return fraction_log_supermodular
