import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from yfinance.exceptions import YFRateLimitError


def _format_tickers(raw_tickers: str) -> list[str]:
    return [t.strip().upper() for t in raw_tickers.replace("\n", ",").split(",") if t.strip()]


def _safe_float(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _safe_get_history(ticker_obj: yf.Ticker) -> pd.DataFrame:
    try:
        return ticker_obj.history(period="3mo", interval="1d")
    except YFRateLimitError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _safe_get_info(ticker_obj: yf.Ticker) -> dict:
    try:
        return ticker_obj.get_info() or {}
    except YFRateLimitError:
        return {}
    except Exception:
        return {}


def _safe_get_fast_info(ticker_obj: yf.Ticker) -> dict:
    try:
        return dict(ticker_obj.fast_info or {})
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(tickers: tuple[str, ...]) -> pd.DataFrame:
    rows = []

    for ticker in tickers:
        tk = yf.Ticker(ticker)
        info = _safe_get_info(tk)
        fast_info = _safe_get_fast_info(tk)
        history = _safe_get_history(tk)

        current_price = np.nan
        mean_50 = np.nan

        if not history.empty and "Close" in history.columns:
            close_series = history["Close"].dropna()
            if not close_series.empty:
                current_price = _safe_float(close_series.iloc[-1])
                mean_50 = _safe_float(close_series.tail(50).mean())

        if np.isnan(current_price):
            current_price = _safe_float(
                fast_info.get("lastPrice", fast_info.get("last_price"))
            )

        if np.isnan(mean_50):
            mean_50 = _safe_float(
                fast_info.get("fiftyDayAverage", fast_info.get("fifty_day_average"))
            )

        pe_ratio = _safe_float(info.get("trailingPE"))
        if np.isnan(pe_ratio):
            pe_ratio = _safe_float(fast_info.get("trailingPE", fast_info.get("trailing_pe")))

        dividend_yield = _safe_float(info.get("dividendYield"))
        if np.isnan(dividend_yield):
            dividend_yield = _safe_float(
                fast_info.get("dividendYield", fast_info.get("dividend_yield"))
            )
        if not np.isnan(dividend_yield):
            # yfinance usually returns dividend yield as decimal (0.02 -> 2.00%).
            if dividend_yield <= 1:
                dividend_yield *= 100
            else:
                # Some providers return percentage already.
                dividend_yield = dividend_yield

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
