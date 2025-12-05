#!/usr/bin/env python3
"""
Compute the simple vibrational proxy σ_mol^(CC) for the C–C, C=C, C≡C
benchmark series used in the σ-locks-in-molecules paper.

Definition:
σ_mol^(CC) = (ν_CC - ν_single) / (ν_triple - ν_single)
"""

def sigma_cc(nu_cc, nu_single, nu_triple):
    return (nu_cc - nu_single) / (nu_triple - nu_single)


def main():
    # Fundamental stretching frequencies in cm^-1 (gas phase, schematic)
    nu_single = 995   # C–C in ethane
    nu_double = 1623  # C=C in ethene
    nu_triple = 1974  # C≡C in ethyne

    bonds = [
        ("C2H6", "C–C",   nu_single),
        ("C2H4", "C=C",   nu_double),
        ("C2H2", "C≡C",   nu_triple),
    ]

    print("Molecule  Bond   nu_stretch/cm^-1   sigma_mol^(CC)")
    print("--------  -----  -----------------   --------------")

    for mol, btype, nu in bonds:
        if btype == "C–C":
            sigma = 0.0
        elif btype == "C≡C":
            sigma = 1.0
        else:
            sigma = sigma_cc(nu, nu_single, nu_triple)
        print(f"{mol:8s}  {btype:5s}  {nu:17.0f}   {sigma:10.2f}")


if __name__ == "__main__":
    main()
