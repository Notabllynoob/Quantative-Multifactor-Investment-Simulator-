from flask import Flask, render_template, request
import os
import pandas as pd
from logic import (
    parse_uploaded_csv,
    get_financial_data,
    perform_clustering,
    run_optimization,
    generate_plot,
    plot_pie_chart,
    plot_esg_distribution,
    format_df_to_html
)

app = Flask(__name__)
app.secret_key = os.urandom(24)


INDEX_DATA = {
    "US Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "CRM"],
    "US Financial & Banking": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP"],
    "International Market ETFs": ["EEM", "EFA", "INDA", "EWG", "MCHI", "EWJ"]
}



@app.route('/')
def index():
    index_names = list(INDEX_DATA.keys())
    return render_template('index.html', indices=index_names)


@app.post('/simulate')
def simulate():

    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')  # get your own alpha vantage api for this and replace this with one in readme
    if not api_key:
        return render_template('results.html', error="ALPHA_VANTAGE_API_KEY environment variable not set.")

    try:

        input_mode = request.form.get('input')
        clustering_method = request.form.get('cluster_method')
        optimization_strategy = request.form.get('optimization_strategy')

        tickers, esg_data = [], None

        if input_mode == 'csv':
            if 'csv_file' not in request.files or not request.files['csv_file'].filename:
                return render_template('results.html', error="No CSV file was uploaded for CSV mode.")
            file = request.files['csv_file']
            tickers, esg_data = parse_uploaded_csv(file.stream)

        elif input_mode == 'picker':
            index_name = request.form.get('stock_index')
            tickers = INDEX_DATA.get(index_name, [])
            esg_data = None  # ESG data is not available in this mode

        if not tickers:
            return render_template('results.html', error="No tickers found. Please check your input.")


        prices, excluded_stocks = get_financial_data(tickers, api_key)

        final_tickers = prices.columns.tolist()
        if esg_data is not None:
            esg_data = esg_data[esg_data['Ticker'].isin(final_tickers)]


        mu = prices.pct_change().mean() * 252
        sigma = prices.pct_change().std() * (252 ** 0.5)
        cluster_df = pd.DataFrame({'Annualized Return': mu, 'Annualized Volatility': sigma})
        clustered_data = perform_clustering(cluster_df.copy(), clustering_method)
        clustered_table = format_df_to_html(clustered_data.reset_index().rename(columns={'index': 'Ticker'}))


        weights = run_optimization(prices, optimization_strategy)
        weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
        weights_df = weights_df[weights_df['Weight'] > 0]
        weights_df['Weight'] = (weights_df['Weight'] * 100).map('{:.2f}%'.format)
        optimized_weights_table = format_df_to_html(weights_df.sort_values(by='Weight', ascending=False))

        pie_chart = generate_plot(plot_pie_chart, weights=weights)

        esg_chart = None
        if esg_data is not None and not esg_data.empty:
            esg_chart = generate_plot(plot_esg_distribution, esg_data=esg_data)


        return render_template(
            'results.html',
            clustering_method=clustering_method,
            optimization_strategy=optimization_strategy,
            excluded_stocks=excluded_stocks,
            optimized_weights_table=optimized_weights_table,
            clustered_table=clustered_table,
            pie_chart=pie_chart,
            esg_chart=esg_chart
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        return render_template('results.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
