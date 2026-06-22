import numpy as np

# D = 0.127851
m = 4
n = 4
p = 4
q = 4
r = 4
u = 4
X0 = 0
alpha1 = 0.8
alpha2 = 0.05
K1 = 2
K2 = 5
# K3 = 2
K4 = 2
K5 = 0.5
K6 = 0.5
alpha3 = 0.05
alpha4 = 0.8
# beta1 = 0.5
beta2 = 2.0
k1 = 0.5
# k2 = 0.6
k3 = 2
k4 = 2
# S0 = 45
# Y_PS1 = 0.4
# Y_XS = 0.6
# Y_PS2 = 1.8
# mu_max = 2.2
K_S = 0.75
K_m = 0.5
    

X_input = np.zeros((1000, 3))

def _activation(value, constant, exponent):
    ratio = (value / constant) ** exponent
    return ratio / (1.0 + ratio)


def _repression(value, constant, exponent):
    ratio = (value / constant) ** exponent
    return 1.0 / (1.0 + ratio)

def MalCoA_switch(t, y, params):
    (
        
        D,
        # alpha1,
        # alpha2,
        # K1,
        # K2,
        K3,
        # K4,
        # K5,
        # K6,
        # alpha3,
        # alpha4,
        beta1,
        # beta2,
        # k1,
        k2,
        # k3,
        # k4,
        S0,
        Y_PS1,
        Y_XS,
        Y_PS2,
        mu_max,
        # K_S,
        # K_m,
    ) = params

    dy = np.zeros(9)

    X = y[0]
    C_FapR = y[1]
    C_FAS = y[2]
    C_ACC = y[3]
    C_FA = y[4]
    C_MalCoA = y[5]
    C_AcCoA = y[6]
    C_PDH = y[7]
    S = y[8]

    mu = mu_max * S / ((K_S + S) * (1.0 + (C_MalCoA / K1)))
    fapr_malcoa_binding = _activation(C_MalCoA, K2, m)
    fas_repression = _repression(C_FapR, K3, n)
    acc_activation = _activation(C_FapR, K4, p)
    fa_production = k2 * C_FAS * _activation(C_MalCoA, K_m, q)
    malcoa_production = k3 * C_ACC * _activation(C_AcCoA, K5, r)
    accoa_production = k4 * C_PDH * _activation(S, K6, u)

    dy[0] = D * (X0 - X) + mu * X
    dy[1] = alpha1 * mu * X - D * C_FapR - k1 * C_FapR * fapr_malcoa_binding
    dy[2] = beta1 * fas_repression - D * C_FAS + alpha2 * mu * X
    dy[3] = beta2 * acc_activation - D * C_ACC + alpha3 * mu * X
    dy[4] = fa_production - D * C_FA
    dy[5] = malcoa_production - fa_production / Y_PS1 - D * C_MalCoA
    dy[6] = accoa_production - malcoa_production - D * C_AcCoA
    dy[7] = alpha4 * mu * X - D * C_PDH
    dy[8] = D * (S0 - S) - accoa_production / Y_PS2 - mu * X / Y_XS

    return dy


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy import integrate 

    # D = 0.13
    # alpha1 = 0.8
    # alpha2 = 0.05
    # K1 = 2
    # K2 = 5
    # K3 = 2
    # K4 = 2
    # K5 = 0.5
    # K6 = 0.5
    # alpha3 = 0.05
    # alpha4 = 0.8
    # beta1 = 0.5
    # beta2 = 2.0
    # k1 = 0.5
    # k2 = 0.6
    # k3 = 2
    # k4 = 2
    # S0 = 45
    # Y_PS1 = 0.4
    # Y_XS = 0.6
    # Y_PS2 = 1.8
    # mu_max = 2.2
    # K_S = 0.75
    # K_m = 0.5

    params = np.array(
        [
        D,
        # alpha1,
        # alpha2,
        # K1,
        # K2,
        K3,
        # K4,
        # K5,
        # K6,
        # alpha3,
        # alpha4,
        beta1,
        # beta2,
        # k1,
        k2,
        # k3,
        # k4,
        S0,
        Y_PS1,
        Y_XS,
        Y_PS2,
        mu_max,
        # K_S,
        # K_m,
        ]
    )

    tspan = (0, 300)
    t_eval = np.arange(0, 300, 1)

    y0 = [1, 1, 1, 1, 0, 1, 1, 1, 1]

    sol = solve_ivp(
        MalCoA_switch,
        tspan,
        y0,
        args=(params,),
        t_eval=t_eval,
        method="BDF",
        rtol=1e-6,
        atol=1e-9,
    )
    
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    plt.figure(figsize=(8, 6))
    plt.plot(sol.t, sol.y[0], "-", linewidth=1.5)
    plt.plot(sol.t, sol.y[1], "-", linewidth=1.5)
    plt.plot(sol.t, sol.y[2], "-", linewidth=1.5)
    plt.plot(sol.t, sol.y[3], "-", linewidth=1.5)
    plt.plot(sol.t, sol.y[4], "-", linewidth=1.5)
    plt.plot(sol.t, sol.y[5], "-", linewidth=1.5)
    plt.plot(sol.t, sol.y[6], "-", linewidth=1.5)
    plt.plot(sol.t, sol.y[7], "-.", linewidth=1.5)
    plt.plot(sol.t, sol.y[8], "-.", linewidth=1.5)
    plt.legend(["Biomass", "FapR", "FAS", "ACC", "FA", "MalCoA", "AcCoA", "PDH", "S"])
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Species concentration (au)", fontsize=16)
    plt.grid(True)
    plt.show()
