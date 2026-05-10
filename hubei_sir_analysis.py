from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


DATA_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_confirmed_global.csv"
)

START_DATE = "2020-01-22"
END_DATE = "2020-04-30"

FIGURE_DIR = Path("figures")
RESULT_DIR = Path("results")


def load_hubei_data():
    confirmed = pd.read_csv(DATA_URL)
    hubei = confirmed[
        (confirmed["Province/State"] == "Hubei")
        & (confirmed["Country/Region"] == "China")
    ]

    if hubei.empty:
        raise ValueError("Hubei row was not found in the Johns Hopkins dataset.")

    date_cols = hubei.columns[4:]
    hubei_long = hubei[date_cols].T.reset_index()
    hubei_long.columns = ["date", "confirmed"]
    hubei_long["date"] = pd.to_datetime(hubei_long["date"], format="%m/%d/%y")
    hubei_long["confirmed"] = hubei_long["confirmed"].astype(float)

    df = hubei_long[
        (hubei_long["date"] >= START_DATE) & (hubei_long["date"] <= END_DATE)
    ].copy()
    df["days"] = (df["date"] - df["date"].iloc[0]).dt.days
    df["new_cases"] = df["confirmed"].diff().fillna(0)
    df["new_cases_7day"] = (
        df["new_cases"].rolling(window=7, center=True, min_periods=1).mean()
    )
    return df


def sir_model(t, y, beta, gamma, n_eff):
    s, i, r = y
    dsdt = -beta * s * i / n_eff
    didt = beta * s * i / n_eff - gamma * i
    drdt = gamma * i
    return [dsdt, didt, drdt]


def beta_logistic_decline(t, beta0, reduction_multiplier, transition_day, width):
    lower_beta = beta0 * reduction_multiplier
    return lower_beta + (beta0 - lower_beta) / (
        1 + np.exp((t - transition_day) / width)
    )


def sir_model_time_varying_beta(
    t, y, beta0, reduction_multiplier, transition_day, width, gamma, n_eff
):
    beta_t = beta_logistic_decline(t, beta0, reduction_multiplier, transition_day, width)
    s, i, r = y
    dsdt = -beta_t * s * i / n_eff
    didt = beta_t * s * i / n_eff - gamma * i
    drdt = gamma * i
    return [dsdt, didt, drdt]


def solve_sir(beta, gamma, n_eff, i0, r0_initial, t_eval):
    s0 = n_eff - i0 - r0_initial
    y0 = [s0, i0, r0_initial]
    sol = solve_ivp(
        fun=lambda t, y: sir_model(t, y, beta, gamma, n_eff),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        max_step=1,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    s, i, r = sol.y
    cumulative_cases = n_eff - s
    daily_new_cases = beta * s * i / n_eff
    return s, i, r, cumulative_cases, daily_new_cases


def solve_sir_time_varying_beta(
    beta0,
    reduction_multiplier,
    transition_day,
    width,
    gamma,
    n_eff,
    i0,
    r0_initial,
    t_eval,
):
    s0 = n_eff - i0 - r0_initial
    y0 = [s0, i0, r0_initial]
    sol = solve_ivp(
        fun=lambda t, y: sir_model_time_varying_beta(
            t, y, beta0, reduction_multiplier, transition_day, width, gamma, n_eff
        ),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        max_step=1,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    s, i, r = sol.y
    beta_t = beta_logistic_decline(t_eval, beta0, reduction_multiplier, transition_day, width)
    cumulative_cases = n_eff - s
    daily_new_cases = beta_t * s * i / n_eff
    return s, i, r, cumulative_cases, daily_new_cases, beta_t


def fit_sir(df):
    t_data = df["days"].to_numpy()
    observed_cumulative = df["confirmed"].to_numpy()
    i0 = observed_cumulative[0]
    r0_initial = 0

    def simulate_cumulative_cases(beta, gamma, n_eff):
        _, _, _, cumulative_cases, _ = solve_sir(
            beta, gamma, n_eff, i0, r0_initial, t_data
        )
        return cumulative_cases

    def residuals(params):
        beta, gamma, n_eff = params
        model_cumulative = simulate_cumulative_cases(beta, gamma, n_eff)
        return (model_cumulative - observed_cumulative) / observed_cumulative.max()

    fit = least_squares(
        residuals,
        x0=[0.35, 0.10, observed_cumulative.max() * 2],
        bounds=(
            [0.01, 0.01, observed_cumulative.max() * 1.01],
            [2.00, 1.00, 5_000_000],
        ),
    )

    beta_fit, gamma_fit, n_eff_fit = fit.x
    return beta_fit, gamma_fit, n_eff_fit, i0, r0_initial, fit.cost


def fit_sir_time_varying_beta(df, beta_guess, gamma_guess, n_eff_guess):
    t_data = df["days"].to_numpy()
    observed_cumulative = df["confirmed"].to_numpy()
    i0 = observed_cumulative[0]
    r0_initial = 0

    def simulate_cumulative_cases(
        beta0, reduction_multiplier, transition_day, width, gamma, n_eff
    ):
        _, _, _, cumulative_cases, _, _ = solve_sir_time_varying_beta(
            beta0,
            reduction_multiplier,
            transition_day,
            width,
            gamma,
            n_eff,
            i0,
            r0_initial,
            t_data,
        )
        return cumulative_cases

    def residuals(params):
        beta0, reduction_multiplier, transition_day, width, gamma, n_eff = params
        model_cumulative = simulate_cumulative_cases(
            beta0, reduction_multiplier, transition_day, width, gamma, n_eff
        )
        return (model_cumulative - observed_cumulative) / observed_cumulative.max()

    fit = least_squares(
        residuals,
        x0=[0.60, 0.50, 25.0, 4.0, 0.40, max(n_eff_guess * 3, 200_000)],
        bounds=(
            [0.01, 0.05, 5.0, 1.0, 0.01, observed_cumulative.max() * 1.01],
            [5.00, 1.00, 45.0, 14.0, 1.00, 5_000_000],
        ),
        max_nfev=1000,
    )

    beta0_fit, reduction_multiplier_fit, transition_day_fit, width_fit, gamma_fit, n_eff_fit = fit.x
    return (
        beta0_fit,
        reduction_multiplier_fit,
        transition_day_fit,
        width_fit,
        gamma_fit,
        n_eff_fit,
        fit.cost,
    )


def build_counterfactual_table(df, beta_fit, gamma_fit, n_eff_fit, i0, r0_initial):
    t_data = df["days"].to_numpy()
    scenarios = {
        "Fitted beta": 1.0,
        "10% lower beta": 0.9,
        "20% lower beta": 0.8,
        "30% lower beta": 0.7,
    }

    rows = []
    curves = {}

    for label, multiplier in scenarios.items():
        beta_scenario = beta_fit * multiplier
        _, _, _, cumulative_cases, daily_new_cases = solve_sir(
            beta_scenario, gamma_fit, n_eff_fit, i0, r0_initial, t_data
        )

        peak_day_index = int(np.argmax(daily_new_cases))
        rows.append(
            {
                "Scenario": label,
                "Beta": beta_scenario,
                "Peak daily cases": daily_new_cases[peak_day_index],
                "Peak date": df["date"].iloc[peak_day_index],
                "Final cumulative cases": cumulative_cases[-1],
            }
        )
        curves[label] = daily_new_cases

    return pd.DataFrame(rows), curves


def save_figures(
    df,
    fitted_cumulative,
    time_varying_cumulative,
    time_varying_beta_t,
    counterfactual_curves,
):
    FIGURE_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["confirmed"])
    plt.xlabel("Date")
    plt.ylabel("Cumulative confirmed cases")
    plt.title("Cumulative Confirmed COVID-19 Cases in Hubei")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "hubei_cumulative_cases.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(df["date"], df["new_cases"], label="Daily new cases")
    plt.plot(
        df["date"],
        df["new_cases_7day"],
        color="black",
        linewidth=2,
        label="7-day rolling average",
    )
    plt.xlabel("Date")
    plt.ylabel("Daily new confirmed cases")
    plt.title("Daily New Confirmed COVID-19 Cases in Hubei")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "hubei_daily_cases.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        df["date"],
        df["confirmed"],
        "o",
        markersize=3,
        label="Observed cumulative cases",
    )
    plt.plot(df["date"], fitted_cumulative, label="Fitted SIR model")
    plt.xlabel("Date")
    plt.ylabel("Cumulative confirmed cases")
    plt.title("Fitted SIR Model for Hubei First COVID-19 Wave")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "hubei_sir_fit.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        df["date"],
        df["confirmed"],
        "o",
        markersize=3,
        label="Observed cumulative cases",
    )
    plt.plot(df["date"], fitted_cumulative, label="Constant beta SIR")
    plt.plot(df["date"], time_varying_cumulative, label="Time-varying beta SIR")
    plt.xlabel("Date")
    plt.ylabel("Cumulative confirmed cases")
    plt.title("Constant vs. Time-Varying Transmission SIR Fit")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "hubei_constant_vs_time_varying_sir_fit.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], time_varying_beta_t)
    plt.xlabel("Date")
    plt.ylabel(r"Transmission rate $\beta(t)$")
    plt.title("Estimated Time-Varying Transmission Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "hubei_time_varying_beta.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    for label, daily_new_cases in counterfactual_curves.items():
        plt.plot(df["date"], daily_new_cases, label=label)
    plt.xlabel("Date")
    plt.ylabel("Model daily new cases")
    plt.title("Counterfactual Transmission-Reduction Scenarios")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "hubei_counterfactual_scenarios.png", dpi=300)
    plt.close()


def main():
    RESULT_DIR.mkdir(exist_ok=True)
    df = load_hubei_data()
    beta_fit, gamma_fit, n_eff_fit, i0, r0_initial, fit_cost = fit_sir(df)

    t_data = df["days"].to_numpy()
    _, _, _, fitted_cumulative, _ = solve_sir(
        beta_fit, gamma_fit, n_eff_fit, i0, r0_initial, t_data
    )

    (
        beta0_tv,
        reduction_multiplier_tv,
        transition_day_tv,
        width_tv,
        gamma_tv,
        n_eff_tv,
        tv_fit_cost,
    ) = (
        fit_sir_time_varying_beta(df, beta_fit, gamma_fit, n_eff_fit)
    )
    _, _, _, time_varying_cumulative, _, time_varying_beta_t = (
        solve_sir_time_varying_beta(
            beta0_tv,
            reduction_multiplier_tv,
            transition_day_tv,
            width_tv,
            gamma_tv,
            n_eff_tv,
            i0,
            r0_initial,
            t_data,
        )
    )

    results_df, counterfactual_curves = build_counterfactual_table(
        df, beta_fit, gamma_fit, n_eff_fit, i0, r0_initial
    )

    parameters_df = pd.DataFrame(
        [
            {
                "beta": beta_fit,
                "gamma": gamma_fit,
                "R0_beta_over_gamma": beta_fit / gamma_fit,
                "N_eff": n_eff_fit,
                "fit_cost": fit_cost,
            }
        ]
    )
    time_varying_parameters_df = pd.DataFrame(
        [
            {
                "beta0": beta0_tv,
                "reduction_multiplier": reduction_multiplier_tv,
                "transition_day": transition_day_tv,
                "transition_date": df["date"].iloc[int(round(transition_day_tv))],
                "transition_width_days": width_tv,
                "gamma": gamma_tv,
                "initial_R0_beta0_over_gamma": beta0_tv / gamma_tv,
                "final_beta": time_varying_beta_t[-1],
                "final_Rt_beta_over_gamma": time_varying_beta_t[-1] / gamma_tv,
                "N_eff": n_eff_tv,
                "fit_cost": tv_fit_cost,
            }
        ]
    )

    df.to_csv(RESULT_DIR / "hubei_first_wave_data.csv", index=False)
    parameters_df.to_csv(RESULT_DIR / "hubei_sir_fit_parameters.csv", index=False)
    time_varying_parameters_df.to_csv(
        RESULT_DIR / "hubei_time_varying_sir_fit_parameters.csv", index=False
    )
    results_df.to_csv(RESULT_DIR / "hubei_sir_scenario_results.csv", index=False)
    save_figures(
        df,
        fitted_cumulative,
        time_varying_cumulative,
        time_varying_beta_t,
        counterfactual_curves,
    )

    print("Constant beta fitted parameters")
    print(parameters_df.to_string(index=False))
    print()
    print("Time-varying beta fitted parameters")
    print(time_varying_parameters_df.to_string(index=False))
    print()
    print("Counterfactual scenarios")
    printable_results = results_df.copy()
    printable_results["Peak date"] = printable_results["Peak date"].dt.strftime("%Y-%m-%d")
    print(printable_results.to_string(index=False))


if __name__ == "__main__":
    main()
