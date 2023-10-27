import torch
import torch.nn as nn
import math

laguerre5_u = torch.tensor(
    [
        0.26356031971814109102031,
        1.41340305910651679221800,
        3.59642577104072208122300,
        7.08581000585883755692200,
        12.6408008442757826594300,
    ]
)

laguerre5_w = torch.tensor(
    [
        0.5217556105828086524759,
        0.3986668110831759274500,
        7.5942449681707595390e-2,
        3.6117586799220484545e-3,
        2.3369972385776227891e-5,
    ]
)


def laguerre5(f, r, a, u=laguerre5_u, w=laguerre5_w):
    k = torch.exp(-r * a) / r
    # u1, w1 = u[0], w[0]
    rinv = 1 / r
    x1 = u[0] * rinv + a
    f1 = f(x1)
    res = w[0] * f1
    for i in range(1, len(u)):
        xi = u[i] * rinv + a
        res += w[i] * f(xi)

    return res * k


def evalpoly(z, *coeffs):
    result = torch.zeros_like(z)
    for coeff in reversed(coeffs):
        # for coeff in coeffs:
        result = result * z + coeff
    return result


class SAFTVRMieConstants:
    def __init__(self):
        self.A = torch.tensor(
            [
                [0.81096, 1.7888, -37.578, 92.284],
                [1.02050, -19.341, 151.26, -463.50],
                [-1.90570, 22.845, -228.14, 973.92],
                [1.08850, -6.1962, 106.98, -677.64],
            ],
            dtype=torch.float64,
        )

        self.phi = torch.tensor(
            [
                [7.5365557, -359.440, 1550.9, -1.199320, -1911.2800, 9236.9],
                [-37.604630, 1825.60, -5070.1, 9.063632, 21390.175, -129430.0],
                [71.745953, -3168.00, 6534.6, -17.94820, -51320.700, 357230.0],
                [-46.835520, 1884.20, -3288.7, 11.34027, 37064.540, -315530.0],
                [-2.4679820, -0.82376, -2.7171, 20.52142, 1103.7420, 1390.2],
                [-0.5027200, -3.19350, 2.0883, -56.63770, -3264.6100, -4518.2],
                [8.0956883, 3.70900, 0.0000, 40.53683, 2556.1810, 4241.6],
            ],
            dtype=torch.float64,
        )

        # c isn't used
        self.c = None


class SAFTVRMieNNParams(nn.Module):
    def __init__(self, Mw, V, T, segment, sigma, lambda_a, lambda_r, epsilon):
        super(SAFTVRMieNNParams, self).__init__()
        # Constants
        self.N_A = torch.tensor(6.02214076e23) 
        self.pi = torch.tensor(math.pi)
        k_B = 1.380649e-23
        self.R = torch.tensor(self.N_A * k_B)
        # Unpack parameters
        self.Mw = Mw
        self.segment = segment
        self.sigma = sigma
        self.lambda_a = lambda_a
        self.lambda_r = lambda_r
        self.epsilon = epsilon
        self.V = V
        self.T = T

        # Initialise derived values
        self.C = self.C_mie(lambda_a, lambda_r)
        self.d = self.d_vrmie(T, sigma, lambda_a, lambda_r, epsilon, self.C)
        self.zetai = self.zeta0123(segment, self.d, V)
        self.rho_S = self.N_A * segment / V
        self.zeta_X, self.sigma3_X = self.zeta_X_sigma3(
            segment, sigma, self.rho_S, self.d
        )
        self.zeta_st = self.sigma3_X * self.rho_S * self.pi / 6

    def d_vrmie(self, T, sigma, lambda_a, lambda_r, epsilon, C):
        Tx = T / epsilon
        theta = C / Tx
        lambda_r_inv = 1.0 / lambda_r
        lambda_a_lambda_r = lambda_a * lambda_r_inv
        f_laguerre = (
            lambda x: x ** (-lambda_r_inv)
            * torch.exp(theta * x ** (lambda_a_lambda_r))
            * lambda_r_inv
            / x
        )
        int_fi = laguerre5(f_laguerre, theta, torch.ones_like(theta))
        d = sigma * (1 - int_fi)
        return d

    def C_mie(self, lambda_a, lambda_r):
        return (lambda_r / (lambda_r - lambda_a)) * torch.pow(
            lambda_r / lambda_a, lambda_a / (lambda_r - lambda_a)
        )

    def zeta0123(self, segment, d, V):
        c = self.pi / 6 * self.N_A * segment / V
        zeta0 = c
        zeta1 = zeta0 * d
        zeta2 = zeta1 * d
        zeta3 = zeta2 * d

        return zeta0, zeta1, zeta2, zeta3

    def zeta_X_sigma3(self, segment, sigma, rho_S, d):
        k_rho_S = rho_S * self.pi / 6 / 8
        zeta_X = k_rho_S * (2 * d) ** 3
        sigma3_X = sigma**3
        # zeta_X = torch.tensor([0.08597839522112394], dtype=torch.float64)

        return zeta_X, sigma3_X


class SAFTVRMieNN(nn.Module):
    def __init__(self):
        super(SAFTVRMieNN, self).__init__()
        self.consts = SAFTVRMieConstants()

    def forward(self, Mw, V, T, params):
        data_in = torch.split(params, 1)
        data = SAFTVRMieNNParams(Mw, V, T, *data_in)
        return self.a_res(data)

    def pressure(self, data: SAFTVRMieNNParams):
        # p = -dA/dV
        def f(V):
            data2 = SAFTVRMieNNParams(
                data.Mw,
                V,
                data.T,
                data.segment,
                data.sigma,
                data.lambda_a,
                data.lambda_r,
                data.epsilon,
            )
            return self.a(data2)

        print(f"V = {V}")
        A = f(data.V)[0]
        print(f"A = {A}")
        dA_dV = torch.autograd.grad(A, data.V, create_graph=True)[0]
        print(f"p = {dA_dV[0]}")
        return -dA_dV

    def a(self, data):
        return data.R * data.T * self.a_ideal(data) + self.a_res(data)

    def a_ideal(self, data):
        #! This is wrong
        # a₀ = A₀/nRT =  ∑(xᵢlog(nxᵢ/V)) - 1 - 1.5log(T)
        # ∑(x .* log.(z/V)) - 1
        a_ideal = -torch.log(data.V) - 1.0 - 1.5 * torch.log(data.T)
        print(f"a_ideal = {a_ideal[0]}")
        return a_ideal

    def a_res(self, data: SAFTVRMieNNParams):
        a_hs = self._a_hs(data)  # * a_hs is correct
        a_disp = self.a_disp(data)
        print(f"a_disp_simple = {a_disp[0]}")
        a_chain = self.a_chain(data)
        print(f"a_chain_simple = {a_chain[0]}")

        a_dispchain = self._a_dispchain(data)
        # print(f"a_res = {(a_hs + a_dispchain)[0]}")
        # return a_hs + a_dispchain
        return a_hs + a_disp + a_chain

    def _a_hs(self, data: SAFTVRMieNNParams):
        return data.segment * self._bmcs_hs(data.zetai)

    # hs
    def _bmcs_hs(self, zetai):
        zeta0, zeta1, zeta2, zeta3 = zetai
        zeta3m1 = 1 - zeta3
        zeta3m1_squared = zeta3m1 * zeta3m1
        return (
            1
            / zeta0
            * (
                3 * zeta1 * zeta2 / zeta3m1
                + zeta2**3 / (zeta3 * zeta3m1_squared)
                + (zeta2**3 / zeta3**2 - zeta0) * torch.log1p(-zeta3)
            )
        )

    def _a_dispchain(self, data: SAFTVRMieNNParams):
        x_S = torch.ones_like(data.Mw)
        C = data.C
        x_0 = data.sigma / data.d
        dij3 = data.d**3
        tau = data.epsilon / data.T

        zeta_st5 = data.zeta_st**5
        zeta_st8 = zeta_st5 * data.zeta_st**3

        # * correct
        KHS, dKHS = self.KHS_fdf(data)

        # * for i in comps (this is pure-component only)
        # Precalculate exponentials of x_0ij
        x0ij_lambda_a = x_0**data.lambda_a
        x0ij_lambda_r = x_0**data.lambda_r
        x0ij_lambda_2a = x_0 ** (2 * data.lambda_a)
        x0ij_lambda_2r = x_0 ** (2 * data.lambda_r)
        x0ij_lambda_ar = x_0 ** (data.lambda_a + data.lambda_r)

        # Calculations for a1 - diagonal
        aS_1_a, daS_1drhoS_a = self._aS_1_fdf(data, data.lambda_a)
        # print(f"aS_1_a = {aS_1_a}")
        aS_1_r, daS_1drhoS_r = self._aS_1_fdf(data, data.lambda_r)
        B_a, dBdrhoS_a = self.B_fdf(data, data.lambda_a, x_0)
        B_r, dBdrhoS_r = self.B_fdf(data, data.lambda_r, x_0)
        a1_ij = (2 * data.pi * data.epsilon * data.d**3) * (
            C
            * data.rho_S
            * (x0ij_lambda_a * (aS_1_a + B_a) - x0ij_lambda_r * (aS_1_r + B_r))
        )

        # Calculations for a2 - diagonal
        aS_1_2a, daS_1drhoS_2a = self._aS_1_fdf(data, 2 * data.lambda_a)
        aS_1_2r, daS_1drhoS_2r = self._aS_1_fdf(data, 2 * data.lambda_r)
        aS_1_ar, daS_1drhoS_ar = self._aS_1_fdf(data, data.lambda_a + data.lambda_r)

        B_2a, dBdrhoS_2a = self.B_fdf(data, 2 * data.lambda_a, x_0)
        B_2r, dBdrhoS_2r = self.B_fdf(data, 2 * data.lambda_r, x_0)
        B_ar, dBdrhoS_ar = self.B_fdf(data, data.lambda_a + data.lambda_r, x_0)

        alpha = C * (1 / (data.lambda_a - 3) - 1 / (data.lambda_r - 3))
        f1, f2, f3, f4, f5, f6 = self.f123456(data, alpha)
        chi = f1 * data.zeta_st + f2 * zeta_st5 + f3 * zeta_st8

        a2_ij = (
            data.pi
            * KHS
            * (1 + chi)
            * data.rho_S
            * data.epsilon**2
            * dij3
            * C**2
            * (
                x0ij_lambda_2a * (aS_1_2a + B_2a)
                - 2 * x0ij_lambda_ar * (aS_1_ar + B_ar)
                + x0ij_lambda_2r * (aS_1_2r + B_2r)
            )
        )

        # Calculations for a3 - diagonal
        # print(f"epsilon = {data.epsilon[0]}")
        # print(f"f4 = {f4}")
        # print(f"f5 = {f5}")
        # print(f"f6 = {f6}")
        # print(f"zeta_st = {data.zeta_st[0]}")
        a3_ij = (
            -data.epsilon**3
            * f4
            * data.zeta_st
            * torch.exp(data.zeta_st * (f5 + f6 * data.zeta_st))
        )

        # Adding - diagnoal
        # * x_Si and x_Sj are both 1 as z * m * m_bar_inv = 1
        # * sum across diagnoal unnecessary as pure component
        a1 = a1_ij  # * Correct
        a2 = a2_ij  #! doesn't match from the 3rd decimal place onwards
        a3 = a3_ij  #! doesn't match from the 4th decimal place onwards
        #         print(
        #             f"""
        # a1 = {a1[0]}
        # a2 = {a2[0]}
        # a3 = {a3[0]}
        #               """
        #         )

        g_HSi = self.g_HS(data, x_0)

        da_1drho_S = C * (
            x0ij_lambda_a * (daS_1drhoS_a + dBdrhoS_a)
            - x0ij_lambda_r * (daS_1drhoS_r + dBdrhoS_r)
        )

        # Calculus for g1
        g_1 = 3 * da_1drho_S - C * (
            data.lambda_a * x0ij_lambda_a * (aS_1_a + B_a)
            - data.lambda_r * x0ij_lambda_r * (aS_1_r + B_r)
        )
        theta = torch.expm1(tau)

        # * gamma_c correct
        gamma_c = (
            10
            * (-torch.tanh(10 * (0.57 - alpha)) + 1)
            * data.zeta_st
            * theta
            * torch.exp(data.zeta_st * (-6.7 - 8 * data.zeta_st))
        )

        da_2drho_S = (
            0.5
            * C**2
            * (
                data.rho_S
                * dKHS
                * (
                    x0ij_lambda_2a * (aS_1_2a + B_2a)
                    - 2 * x0ij_lambda_ar * (aS_1_ar + B_ar)
                    + x0ij_lambda_2r * (aS_1_2r + B_2r)
                )
                + KHS
                * (
                    x0ij_lambda_2a * (daS_1drhoS_2a + dBdrhoS_2a)
                    - 2 * x0ij_lambda_ar * (daS_1drhoS_ar + dBdrhoS_ar)
                    + x0ij_lambda_2r * (daS_1drhoS_2r + dBdrhoS_2r)
                )
            )
        )
        # print(f"da_2drho_S = {da_2drho_S[0]}")

        gMCA2 = 3 * da_2drho_S - KHS * C**2 * (
            data.lambda_r * x0ij_lambda_2r * (aS_1_2r + B_2r)
            - (data.lambda_a + data.lambda_r) * x0ij_lambda_ar * (aS_1_ar + B_ar)
            + data.lambda_a * x0ij_lambda_2a * (aS_1_2a + B_2a)
        )

        g_2 = (1 + gamma_c) * gMCA2
        g_Mie = g_HSi * torch.exp(tau * g_1 / g_HSi + tau**2 * g_2 / g_HSi)
        achain = -torch.log(g_Mie) * (data.segment - 1)

        a1 = a1 * data.segment / T
        a2 = a2 * data.segment / T**2
        a3 = a3 * data.segment / T**3
        adisp = a1 + a2 + a3

        print(f"adisp = {adisp[0]}")
        print(f"achain = {achain[0]}")
        return adisp + achain

    def _aS_1_fdf(self, data: SAFTVRMieNNParams, lambda_i):
        zeta_eff, dzeta_eff = self._zeta_eff_fdf(data, lambda_i)
        # print(f"zeta_eff = {zeta_eff}")
        zeta_eff3 = (1 - zeta_eff) ** 3
        zeta_effm1 = 1 - zeta_eff * 0.5
        zeta_f = zeta_effm1 / zeta_eff3
        lambda_f = -1 / (lambda_i - 3)
        _f = lambda_f * zeta_f
        _df = lambda_f * (
            zeta_f
            + data.rho_S
            * dzeta_eff
            * (
                (3 * zeta_effm1 * (1 - zeta_eff) ** 2 - 0.5 * zeta_eff3)
                / (zeta_eff3**2)
            )
        )
        # print(f"_f = {_f}")
        # print(f"_df = {_df}")
        return _f, _df

    def _zeta_eff_fdf(self, data: SAFTVRMieNNParams, lambda_i):
        A = self.consts.A
        lambda_inv = torch.ones_like(lambda_i) / lambda_i

        # Element-wise multiply A with each column of lambda_inv_powers and sum along the last dimension
        A_lambda_inv = torch.sum(
            A
            * torch.tensor(
                [
                    torch.ones_like(lambda_inv),
                    lambda_inv,
                    lambda_inv**2,
                    lambda_inv**3,
                ]
            ),
            dim=1,
        )

        zeta_X = data.zeta_X
        rho_S = data.rho_S
        f = torch.dot(
            A_lambda_inv,
            torch.tensor([zeta_X, zeta_X**2, zeta_X**3, zeta_X**4]),
        )

        df = (
            torch.dot(
                A_lambda_inv,
                torch.tensor(
                    [
                        torch.ones_like(lambda_inv),
                        2 * zeta_X,
                        3 * zeta_X**2,
                        4 * zeta_X**3,
                    ]
                ),
            )
            * zeta_X
            / rho_S
        )

        return f, df

    def B_fdf(self, data: SAFTVRMieNNParams, lambda_i, x_0):
        x_0_lambda = x_0 ** (3 - lambda_i)
        # * I, J correct
        I = (1 - x_0_lambda) / (lambda_i - 3)
        J = (
            1 - (lambda_i - 3) * x_0 ** (4 - lambda_i) + (lambda_i - 4) * x_0_lambda
        ) / ((lambda_i - 3) * (lambda_i - 4))

        zeta_X2 = (1 - data.zeta_X) ** 2
        zeta_X3 = (1 - data.zeta_X) ** 3
        zeta_X6 = zeta_X3 * zeta_X2

        # print(f"zeta_X = {data.zeta_X[0]}")
        # print(f"I = {I[0]}")
        # print(f"J = {J[0]}")

        f = I * (1 - data.zeta_X / 2) / zeta_X3 - 9 * J * data.zeta_X * (
            data.zeta_X + 1
        ) / (2 * zeta_X3)
        #! bug here/!?!?!?
        # df = (
        #     (1 - data.zeta_X / 2) * I / zeta_X3
        #     - 9 * data.zeta_X * (1 + data.zeta_X) * J / (2 * zeta_X3)
        #     + data.zeta_X
        #     * (
        #         (3 * (1 - data.zeta_X / 2) * zeta_X2 - 1 / 2 * zeta_X3) * I / zeta_X6
        #         - 9
        #         * J
        #         * (
        #             (1 + 2 * data.zeta_X) * zeta_X3
        #             + data.zeta_X * (1 + data.zeta_X) * 3 * zeta_X2
        #         )
        #         / (2 * zeta_X6)
        #     )
        # )
        term1 = (1 - data.zeta_X / 2) * I / zeta_X3 - 9 * data.zeta_X * (
            1 + data.zeta_X
        ) * J / (2 * zeta_X3)
        term2 = data.zeta_X * (
            (3 * (1 - data.zeta_X / 2) * zeta_X2 - 0.5 * zeta_X3) * I / zeta_X6
            - 9
            * J
            * (
                (1 + 2 * data.zeta_X) * zeta_X3
                + data.zeta_X * (1 + data.zeta_X) * 3 * zeta_X2
            )
            / (2 * zeta_X6)
        )

        df = term1 + term2
        # print(f"df = {df}")

        return f, df

    def f123456(self, data: SAFTVRMieNNParams, alpha):
        phi = self.consts.phi

        fa = torch.zeros(6)
        fb = torch.zeros(6)

        for i in range(4):
            fa += phi[i] * alpha**i

        for i in range(4, 7):
            fb += phi[i] * alpha ** (i - 3)

        return fa / (1 + fb)

    def KHS_fdf(self, data: SAFTVRMieNNParams):
        zeta_X4 = (1 - data.zeta_X) ** 4
        denom1 = evalpoly(data.zeta_X, 1, 4, 4, -4, 1)
        ddenom1 = evalpoly(data.zeta_X, 4, 8, -12, 4)
        f = zeta_X4 / denom1
        df = -(data.zeta_X / data.rho_S) * (
            (4 * (1 - data.zeta_X) ** 3 * denom1 + zeta_X4 * ddenom1) / denom1**2
        )
        return f, df

    def g_HS(self, data: SAFTVRMieNNParams, x_0):
        zeta_X3 = (1 - data.zeta_X) ** 3
        k_0 = -torch.log(1 - data.zeta_X) + evalpoly(data.zeta_X, 0, 42, -39, 9, -2) / (
            6 * zeta_X3
        )
        k_1 = evalpoly(data.zeta_X, 0, -12, 6, 0, 1) / (2 * zeta_X3)
        k_2 = -3 * data.zeta_X**2 / (8 * (1 - data.zeta_X) ** 2)
        k_3 = evalpoly(data.zeta_X, 0, 3, 3, 0, -1) / (6 * zeta_X3)

        return torch.exp(evalpoly(x_0, k_0, k_1, k_2, k_3))

    #! Implementing for validation
    def a_disp(self, data: SAFTVRMieNNParams):
        zeta_st5 = data.zeta_st**5
        zeta_st8 = zeta_st5 * data.zeta_st**3
        # KHS = self.KHS(data)
        KHS = (1 - data.zeta_X) ** 4 / (
            1
            + 4 * data.zeta_X
            + 4 * data.zeta_X**2
            - 4 * data.zeta_X**3
            + data.zeta_X**4
        )
        C = data.C
        d3 = data.d**3
        x0 = data.sigma / data.d

        aS_1_a = self.aS_1(data, data.lambda_a)
        aS_1_r = self.aS_1(data, data.lambda_r)

        B_a = self.B(data, data.lambda_a, x0)
        B_r = self.B(data, data.lambda_r, x0)

        a1 = (
            (2 * data.pi * data.epsilon * d3)
            * C
            * data.rho_S
            * (
                x0**data.lambda_a * (aS_1_a + B_a)
                - x0**data.lambda_r * (aS_1_r + B_r)
            )
        )

        alpha = C * (1 / (data.lambda_a - 3) - 1 / (data.lambda_r - 3))
        f1, f2, f3, f4, f5, f6 = self.f123456(data, alpha)
        chi = f1 * data.zeta_st + f2 * zeta_st5 + f3 * zeta_st8

        f = lambda x: x0 ** (x) * self.aS_1(data, x) * self.B(data, x, x0)
        a2 = (
            data.pi
            * KHS
            * (1 + chi)
            * data.rho_S
            * data.epsilon**2
            * d3
            * C**2
            * (
                f(2 * data.lambda_a)
                - 2 * f(data.lambda_a + data.lambda_r)
                + f(2 * data.lambda_r)
            )
        )

        a3 = (
            -data.epsilon**3
            * f4
            * data.zeta_st
            * torch.exp(f5 * data.zeta_st + f6 * data.zeta_st**2)
        )

        adisp = data.segment * (a1 / data.T + a2 / data.T**2 + a3 / data.T**3)

        return adisp

    def aS_1(self, data, lambda_i):
        zeta_eff = self.zeta_eff(data, lambda_i)
        return -1 / (lambda_i - 3) * (1 - zeta_eff * 0.5) / (1 - zeta_eff) ** 3

    def B(self, data, lambda_i, x0):
        x0_3lambda = x0 ** (3 - lambda_i)
        zeta_X_m13 = (1 - data.zeta_X) ** 3
        I = (1 - x0_3lambda) / (lambda_i - 3)
        J = (
            1 - (lambda_i - 3) * x0 ** (4 - lambda_i) + (lambda_i - 4) * x0_3lambda
        ) / ((lambda_i - 3) * (lambda_i - 4))
        return I * (1 - data.zeta_X / 2) / zeta_X_m13 - 9 * J * data.zeta_X * (
            data.zeta_X + 1
        ) / (2 * zeta_X_m13)

    def zeta_eff(self, data, lambda_i):
        A = self.consts.A
        _1 = torch.ones_like(lambda_i)
        lambda_inv = _1 / lambda_i
        zeta_X = data.zeta_X
        lambda_powers = torch.tensor([_1, lambda_inv, lambda_inv**2, lambda_inv**3])

        A_lambda_inv = torch.mv(A, lambda_powers)

        zeta_powers = torch.tensor([zeta_X, zeta_X**2, zeta_X**3, zeta_X**4])

        return torch.dot(A_lambda_inv, zeta_powers)
        # A_lambda_inv = A * torch.tensor(
        #     [
        #         torch.ones_like(lambda_inv),
        #         lambda_inv,
        #         lambda_inv**2,
        #         lambda_inv**3,
        #     ]
        # )

        # return torch.dot(
        #     A_lambda_inv,
        #     torch.tensor(
        #         [data.zeta_X, data.zeta_X**2, data.zeta_X**3, data.zeta_X**4]
        #     ),
        # )

    def a_chain(self, data):
        # zeta_st5 = data.zeta_st**5
        # zeta_st8 = zeta_st5 * data.zeta_st**3
        KHS, dKHS = self.KHS_fdf(data)

        C = data.C
        x0 = data.sigma / data.d
        # d3 = data.d**3

        aS_1_a, daS_1drhoS_a = self._aS_1_fdf(data, data.lambda_a)
        aS_1_r, daS_1drhoS_r = self._aS_1_fdf(data, data.lambda_r)

        B_a, dBdrhoS_a = self.B_fdf(data, data.lambda_a, x0)
        B_r, dBdrhoS_r = self.B_fdf(data, data.lambda_r, x0)

        aS_1_2a, daS_1drhoS_2a = self._aS_1_fdf(data, 2 * data.lambda_a)
        aS_1_2r, daS_1drhoS_2r = self._aS_1_fdf(data, 2 * data.lambda_r)
        aS_1_ar, daS_1drhoS_ar = self._aS_1_fdf(data, data.lambda_a + data.lambda_r)

        B_2a, dBdrhoS_2a = self.B_fdf(data, 2 * data.lambda_a, x0)
        B_2r, dBdrhoS_2r = self.B_fdf(data, 2 * data.lambda_r, x0)
        B_ar, dBdrhoS_ar = self.B_fdf(data, data.lambda_a + data.lambda_r, x0)

        alpha = C * (1 / (data.lambda_a - 3) - 1 / (data.lambda_r - 3))

        g_HSi = self.g_HS(data, x0)
        da_1drho_S = C * (
            x0**data.lambda_a * (daS_1drhoS_a + dBdrhoS_a)
            - x0**data.lambda_r * (daS_1drhoS_r + dBdrhoS_r)
        )

        g_1 = 3 * da_1drho_S - C * (
            data.lambda_a * x0**data.lambda_a * (aS_1_a + B_a)
            - data.lambda_r * x0**data.lambda_r * (aS_1_r + B_r)
        )
        theta = torch.expm1(data.epsilon / data.T)
        gamma_c = (
            10
            * (-torch.tanh(10 * (0.57 - alpha)) + 1)
            * data.zeta_st
            * theta
            * torch.exp(data.zeta_st * (-6.7 - 8 * data.zeta_st))
        )

        da_2drho_S = (
            0.5
            * C**2
            * (
                data.rho_S
                * dKHS
                * (
                    x0 ** (2 * data.lambda_a) * (aS_1_2a + B_2a)
                    - 2 * x0 ** (data.lambda_a + data.lambda_r) * (aS_1_ar + B_ar)
                    + x0 ** (2 * data.lambda_r) * (aS_1_2r + B_2r)
                )
                + KHS
                * (
                    x0 ** (2 * data.lambda_a) * (daS_1drhoS_2a + dBdrhoS_2a)
                    - 2
                    * x0 ** (data.lambda_a + data.lambda_r)
                    * (daS_1drhoS_ar + dBdrhoS_ar)
                    + x0 ** (2 * data.lambda_r) * (daS_1drhoS_2r + dBdrhoS_2r)
                )
            )
        )

        gMCA2 = 3 * da_2drho_S - KHS * C**2 * (
            data.lambda_r * x0 ** (2 * data.lambda_r) * (aS_1_2r + B_2r)
            - (data.lambda_a + data.lambda_r)
            * x0 ** (data.lambda_a + data.lambda_r)
            * (aS_1_ar + B_ar)
            + data.lambda_a * x0 ** (2 * data.lambda_a) * (aS_1_2a + B_2a)
        )

        g_2 = (1 + gamma_c) * gMCA2
        g_Mie = g_HSi * torch.exp(
            data.epsilon / data.T * g_1 / g_HSi
            + (data.epsilon / data.T) ** 2 * g_2 / g_HSi
        )
        achain = -torch.log(g_Mie) * (data.segment - 1)
        print(f"achain = {achain[0]}")
        return achain


if __name__ == "__main__":
    if True:
        Mw = torch.tensor([16.04], dtype=torch.float64)
        segment = torch.tensor([1.0], dtype=torch.float64)
        sigma = torch.tensor([3.737e-10], dtype=torch.float64)
        lambda_a = torch.tensor([6.0], dtype=torch.float64)
        lambda_r = torch.tensor([12.504], dtype=torch.float64)
        epsilon = torch.tensor([152.58], dtype=torch.float64)

        V = torch.tensor([1e-2], dtype=torch.float64, requires_grad=True)
        T = torch.tensor([400.0], dtype=torch.float64, requires_grad=True)
        data = SAFTVRMieNNParams(Mw, V, T, segment, sigma, lambda_a, lambda_r, epsilon)
        #! Very large & small numbers in data mean Float32s are not precise enough
        #! Need to either re-work function or take training performance penalty
        saft = SAFTVRMieNN()
        a_res = saft.a_res(data)
        p = saft.pressure(data)
        print(f"methane a_res = {a_res[0]}")
        print(f"methane p = {p[0]}")

    if False:
        # * Test for decane -> need to check chain term
        Mw = torch.tensor([142.29], dtype=torch.float64)
        segment = torch.tensor([2.9976], dtype=torch.float64)
        sigma = torch.tensor([4.589e-10], dtype=torch.float64)
        lambda_a = torch.tensor([6.0], dtype=torch.float64)
        lambda_r = torch.tensor([18.885], dtype=torch.float64)
        epsilon = torch.tensor([400.79], dtype=torch.float64)

        data = SAFTVRMieNNParams(Mw, V, T, segment, sigma, lambda_a, lambda_r, epsilon)
        saft = SAFTVRMieNN()

        a_res = saft.a_res(data)
        p = saft.pressure(data)
        print(f"decane a_res = {a_res[0]}")
        print(f"decane p = {p[0]}")

    # * Test if differentiable

    # model = SAFTVRMieNN()
    # X = torch.tensor(
        # [2.9976, 4.589e-10, 6.0, 18.885, 400.79],
        # dtype=torch.float64,
        # requires_grad=True,
    # )  # .reshape(1, -1)
    # print(f"X = {X}")
    # output = model(Mw, V, T, X)
    # dadX = torch.autograd.grad(output, X, create_graph=True)

    # print(f"dadX = {dadX}")

    # * Evaluate both at the same time
    # Mw = torch.tensor([16.04, 142.29], dtype=torch.float64)
    # segment = torch.tensor([1.0, 2.9976], dtype=torch.float64)
    # sigma = torch.tensor([3.737e-10, 4.589e-10], dtype=torch.float64)
    # lambda_a = torch.tensor([6.0, 6.0], dtype=torch.float64)
    # lambda_r = torch.tensor([12.504, 18.885], dtype=torch.float64)
    # epsilon = torch.tensor([152.58, 400.79], dtype=torch.float64)
    # data = SAFTVRMieNNParams(Mw, segment, sigma, lambda_a, lambda_r, epsilon, V, T)
    # saft = SAFTVRMieNN()

    # a_res = saft.a_res(data)
    # print(f"methane a_res = {a_res[0]}, decane a_res = {a_res[1]}")
