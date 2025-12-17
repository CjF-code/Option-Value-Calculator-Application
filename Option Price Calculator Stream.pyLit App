import streamlit as st
import black_scholes as bs
import numpy as np
import matplotlib.pyplot as plt

if not st.session_state:
    st.session_state["counter"] = 0
    st.session_state["chosen_option_dict"] = {}

with st.sidebar:
    st.header("Black Scholes Paramater Input:")
    r = st.number_input(
        "Risk Free Rate", max_value=1.0, min_value=0.0, format="%0.3f", value=0.01
    )
    S = st.number_input(
        "Spot Price", step=1.0, min_value=0.01, format="%0.2f", value=30.0
    )
    K = st.number_input(
        "Strike Price", step=1.0, min_value=0.01, format="%0.2f", value=40.0
    )
    T = st.number_input("Time to maturity (Days)", min_value=0.1, step=1.0, value=240.0)
    sigma = st.number_input("Volatility", min_value=0.001, format="%0.3f", value=0.3)

    # Calculate option prices
    call_option_price = bs.blackScholes(
        r, S, K, T / 365.25, sigma, type="c"
    )
    put_option_price = bs.blackScholes(
        r, S, K, T / 365.25, sigma, type="p"
    )

    # Columns for the buttons to save the call and put options so that they can be used in the payoff display
    col1, col2 = st.columns([1, 1])
    with col1:
        # Sets a limit of 4 saved options
        if (
            st.button("Add Call", use_container_width=True)
            and len(st.session_state.chosen_option_dict.keys()) < 4
        ):
            # Creating the Call option in the saved ptions dict
            option_features = {
                "Type": "Call",
                "Duration": int(T),
                "Premium": round(call_option_price, 4),
                "Strike": K,
                "Spot": S,
                "Long": 0,
                "Short": 0,
            }
            st.session_state.chosen_option_dict[
                f"{option_features['Type']} | {round(option_features['Premium'],2)} | {option_features['Strike']}"
            ] = option_features

    with col2:
        # Sets a limit of 4 saved options
        if (
            st.button("Add Put", use_container_width=True)
            and len(st.session_state.chosen_option_dict.keys()) < 4
        ):
            # Creating the Put option in the saved ptions dict
            option_features = {
                "Type": "Put",
                "Duration": int(T),
                "Premium": round(put_option_price, 4),
                "Strike": K,
                "Spot": S,
                "Long": 0,
                "Short": 0,
            }
            st.session_state.chosen_option_dict[
                f"{option_features['Type']} | {round(option_features['Premium'],2)} | {option_features['Strike']}"
            ] = option_features

    if len(st.session_state.chosen_option_dict.keys()) == 4:
        st.warning("Max Number of Options Saved.")


st.subheader("Theoretical Black-Scholes Option Prices Using Side Bar Parameters:")

# Black-Scholes Calculated values display
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown(
        f"""
        <div style="background-color: #6dbf53; padding: 20px; border-radius: 10px; text-align: center;">
            <span style="color: white; font-size: 24px; font-weight: bold;">Call: ${call_option_price:.2f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div style="background-color: #5c99d9; padding: 20px; border-radius: 10px; text-align: center;">
            <span style="color: white; font-size: 24px; font-weight: bold;">Put: ${put_option_price:.2f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

st.markdown("")
st.markdown("")
st.subheader("Saved Options (Max. 4)")

# Display for key values from dictionary for the saved options
col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
with col1:
    st.write("**Option Name**")
    for option in st.session_state.chosen_option_dict.keys():
        st.write(option)
with col2:
    st.write("**Duration**")
    for option in st.session_state.chosen_option_dict.keys():
        st.write(f"{st.session_state.chosen_option_dict[option]['Duration']} days")

with col3:
    st.write("**Premium**")
    for option in st.session_state.chosen_option_dict.keys():
        st.write(f"${st.session_state.chosen_option_dict[option]['Premium']}")

with col4:
    st.write("**Strike**")
    for option in st.session_state.chosen_option_dict.keys():
        st.write(f"${st.session_state.chosen_option_dict[option]['Strike']}")

with col5:
    st.write("**Spot Price**")
    for option in st.session_state.chosen_option_dict.keys():
        st.write(f"${st.session_state.chosen_option_dict[option]['Spot']:.2f}")

st.markdown("")
st.markdown("")

st.write("**Delete Options**")

# Variable to save name of option to be deleted
option_to_delete = st.selectbox(
    label="Select an Option to delete",
    options=st.session_state.chosen_option_dict.keys(),
)

# Button to delete options from "saved options" sectoin
if st.button("Delete Selected Option 🗑️", use_container_width=True):
    if len(st.session_state.chosen_option_dict.keys()) > 0:
        del st.session_state.chosen_option_dict[option_to_delete]
    st.rerun()

# Form layout
with st.form("Chose Option Positions"):
    # Dynamic columns for display of long or short of selected options
    if st.session_state.chosen_option_dict.keys():
        num_columns = len(st.session_state.chosen_option_dict)
        cols = st.columns(num_columns)

        for idx, (option, positions) in enumerate(
            st.session_state.chosen_option_dict.items()
        ):
            col = cols[idx]
            with col:
                st.write(f"**{option}**")
                buy_col, sell_col = st.columns(2)

                with buy_col:
                    positions["Long"] = st.number_input(
                        key=f"long_{option}_{idx}",
                        label="No. Long",
                        value=positions["Long"],
                        step=1,
                        max_value=4,
                        min_value=0,
                        placeholder=0,
                    )

                with sell_col:
                    positions["Short"] = st.number_input(
                        key=f"short_{option}_{idx}",
                        label="No. Short",
                        value=positions["Short"],
                        step=1,
                        max_value=4,
                        min_value=0,
                        placeholder=0,
                    )

        # Static columns for long or short underlying share
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.long_share = st.number_input(
                key=f"Long Share",
                label="Long Share",
                value=0,
                step=1,
                max_value=4,
                min_value=0,
                placeholder=0,
            )
        with col2:
            st.session_state.short_share = st.number_input(
                key=f"Short Share",
                label="Short Share",
                value=0,
                step=1,
                max_value=4,
                min_value=0,
                placeholder=0,
            )

        if st.form_submit_button("Save Positions"):
            # Save the updated dictionary to session state
            st.session_state.chosen_option_dict = st.session_state.chosen_option_dict

# Calcs and display of payoffs based on chosed option and share strategy saved in the chosen_option_dict in the session state
if len(st.session_state.chosen_option_dict.keys()) > 0:
    # 1. Extract the values from session state
    premiums = [
        option["Premium"] for option in st.session_state.chosen_option_dict.values()
    ]
    strikes = [
        option["Strike"] for option in st.session_state.chosen_option_dict.values()
    ]
    longs = [option["Long"] for option in st.session_state.chosen_option_dict.values()]
    shorts = [
        option["Short"] for option in st.session_state.chosen_option_dict.values()
    ]
    option_types = [
        option["Type"] for option in st.session_state.chosen_option_dict.values()
    ]

    # 2. Define the bounds for the x-axis (spot prices)
    min_strike = min(strikes)
    max_strike = max(strikes)
    buffer_percentage = 0.50
    min_bound = min(
        np.floor(min_strike * buffer_percentage), S * buffer_percentage
    )
    max_bound = max(
        np.ceil(max_strike * (1 + buffer_percentage)),
        S * (1 + buffer_percentage),
    )
    x_vals = np.arange(min_bound, max_bound + 0.01, 0.01)

    # 3. Initialize the total payoff array
    payoffs = np.zeros_like(x_vals)

    # 4a. Calculate payoffs for holdings of underlying share
    if st.session_state["long_share"] > 0:
        share_payoff = st.session_state["long_share"] * (x_vals - S)
        payoffs += share_payoff

    if st.session_state["short_share"] > 0:
        share_payoff = st.session_state["short_share"] * (S - x_vals)
        payoffs += share_payoff

    # 4b. Calculate payoffs for each option
    for i in range(len(strikes)):
        option_type = option_types[i]
        strike = strikes[i]
        premium = premiums[i]
        long = longs[i]
        short = shorts[i]

        if option_type == "Call":
            if long > 0:
                payoffs += long * bs.call_payoff(x_vals, strike, premium, "long")
            if short > 0:
                payoffs += short * bs.call_payoff(x_vals, strike, premium, "short")
        else:  # Put option
            if long > 0:
                payoffs += long * bs.put_payoff(x_vals, strike, premium, "long")
            if short > 0:
                payoffs += short * bs.put_payoff(x_vals, strike, premium, "short")

    # 5. Plotting of calculated payoffs over range of xvals
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals, payoffs, label="Total Payoff", color="blue")

    # Add breakeven points
    breakeven_points = x_vals[np.where(np.diff(np.signbit(payoffs)))[0]]
    for point in breakeven_points:
        ax.axvline(x=point, color="red", linestyle="--", alpha=0.3)
        ax.text(
            point,
            ax.get_ylim()[0],
            f"BE: {point:.2f}",
            rotation=90,
            verticalalignment="bottom",
        )

    # Add styling
    ax.set_title("Options Strategy Payoff Diagram", fontsize=14)
    ax.set_xlabel("Underlying Price", fontsize=12)
    ax.set_ylabel("Profit/Loss", fontsize=12)
    ax.axhline(0, color="black", linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add maximum profit/loss annotations
    max_profit = np.max(payoffs)
    min_loss = np.min(payoffs)
    ax.text(
        0.02,
        0.98,
        f"Max Profit: ${max_profit:.2f}\nMax Loss: ${min_loss:.2f}",
        transform=ax.transAxes,
        verticalalignment="top",
    )

    st.pyplot(fig)
    st.write(
        "*Max Profit and Max Loss are only valid for values shown in the x-axis, a greater profit or loss may be possible outside of the dislayed values.*"
    )
