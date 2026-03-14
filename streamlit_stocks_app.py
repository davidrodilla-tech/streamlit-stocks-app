import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def _format_tickers(raw_tickers: str) -> list[str]:
    return [t.strip().upper() for t in raw_tickers.replace("\n", ",").split(",") if t.strip()]


def _safe_float(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(tickers: tuple[str, ...]) -> pd.DataFrame:
    rows = []

    for ticker in tickers:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        history = tk.history(period="3mo", interval="1d")

        if history.empty or "Close" not in history.columns:
            rows.append(
                {
                    "Ticker": ticker,
                    "Precio actual": np.nan,
                    "P/E Ratio": np.nan,
                    "Dividend Yield (%)": np.nan,
                    "Media 50 sesiones": np.nan,
                    "Debajo media 50": False,
                }
            )
            continue

        close_series = history["Close"].dropna()
        if close_series.empty:
            current_price = np.nan
            mean_50 = np.nan
        else:
            current_price = _safe_float(close_series.iloc[-1])
            mean_50 = _safe_float(close_series.tail(50).mean())

        pe_ratio = _safe_float(info.get("trailingPE"))
        dividend_yield = _safe_float(info.get("dividendYield"))
        if not np.isnan(dividend_yield):
            dividend_yield *= 100

        below_50 = bool(
            not np.isnan(current_price)
            and not np.isnan(mean_50)
            and current_price < mean_50
        )

        rows.append(
            {
                "Ticker": ticker,
                "Precio actual": current_price,
                "P/E Ratio": pe_ratio,
                "Dividend Yield (%)": dividend_yield,
                "Media 50 sesiones": mean_50,
                "Debajo media 50": below_50,
            }
        )

    df = pd.DataFrame(rows)
    return df


def main():
    st.set_page_config(page_title="Analizador de Tickers", page_icon="📈", layout="wide")

    st.title("📈 Analizador de acciones con yfinance")
    st.write(
        "Ingresa una lista de tickers (por ejemplo: `AAPL, TSLA, MSFT`) para ver precio actual, "
        "P/E Ratio y Dividend Yield."
    )

    raw_input = st.text_area(
        "Tickers",
        value="AAPL, TSLA, MSFT",
        help="Separa por comas o saltos de línea",
    )

    only_below_mean = st.checkbox("Mostrar solo acciones por debajo de la media de 50 sesiones")

    if st.button("Consultar"):
        tickers_list = _format_tickers(raw_input)

        if not tickers_list:
            st.warning("Ingresa al menos un ticker válido.")
            st.stop()

        with st.spinner("Descargando datos..."):
            data = get_stock_data(tuple(tickers_list))

        if only_below_mean:
            data = data[data["Debajo media 50"]]

        if data.empty:
            st.info("No hay resultados para mostrar con el filtro seleccionado.")
            st.stop()

        st.subheader("Resultados")

        def highlight_below_mean(row):
            if row["Debajo media 50"]:
                return ["background-color: #c6f6d5"] * len(row)
            return [""] * len(row)

        display_columns = [
            "Ticker",
            "Precio actual",
            "P/E Ratio",
            "Dividend Yield (%)",
            "Media 50 sesiones",
            "Debajo media 50",
        ]

        styled = (
            data[display_columns]
            .sort_values("Ticker")
            .style.apply(highlight_below_mean, axis=1)
            .format(
                {
                    "Precio actual": "{:,.2f}",
                    "P/E Ratio": "{:,.2f}",
                    "Dividend Yield (%)": "{:,.2f}",
                    "Media 50 sesiones": "{:,.2f}",
                },
                na_rep="N/D",
            )
        )

        st.dataframe(styled, use_container_width=True)
        st.caption("Filas en verde: precio actual por debajo de la media de cierre de 50 sesiones.")


if __name__ == "__main__":
    main()
