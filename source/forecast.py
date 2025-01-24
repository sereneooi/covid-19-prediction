import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def show_trends(title, x_variable, y_variable):
    """Display trends using a line graph."""
    plt.title(title)
    plt.plot(x_variable, y_variable)
    x_axis = plt.gca().xaxis
    x_axis.set_major_locator(mdates.MonthLocator())
    x_axis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    plt.ylabel("Cases")
    plt.show()

def calculate_vaccine_type_for_death():
    """Calculate the number of deaths for each vaccine type."""
    deaths = pd.read_csv("linelist_deaths.csv")
    deaths.fillna(value='NaN', inplace=True)

    vaccine_counts = {
        'pfizer': 0,
        'sinovac': 0,
        'astrazeneca': 0,
        'cansino': 0
    }

    for i in range(len(deaths)):
        brand = deaths['brand1'].iloc[i].lower()
        if brand in vaccine_counts:
            vaccine_counts[brand] += 1

    return vaccine_counts['pfizer'], vaccine_counts['sinovac'], vaccine_counts['astrazeneca'], vaccine_counts['cansino']

def calculate_vax_death_rate(vax_type_count, total_vaccinated):
    """Calculate the death rate for a specific vaccine type."""
    return (vax_type_count / total_vaccinated) * 100

def calculate_vaccine_type(vax_type):
    """Calculate the total number vaccinated for a specific vaccine type."""
    vax_type.fillna(value=0, inplace=True)
    return vax_type.sum()

def predict_deaths(data, p_death_rate, s_death_rate, az_death_rate):
    """Predict deaths based on vaccine types and death rates."""
    new_data = data[['cases_new', 'deaths_new', 'pfizer1', 'sinovac1', 'astra1']].copy()

    new_data['predicted_pfizer'] = new_data['pfizer1'] * p_death_rate
    new_data['predicted_sinovac'] = new_data['sinovac1'] * s_death_rate
    new_data['predicted_astrazeneca'] = new_data['astra1'] * az_death_rate
    new_data['total_predicted'] = (
        new_data['predicted_pfizer'] + new_data['predicted_sinovac'] + new_data['predicted_astrazeneca']
    )

    plt.title("Predicted vs Actual Cases")
    new_data['total_predicted'].plot(label='Predicted', color='red')
    new_data['cases_new'].plot(label='Actual Cases', color='blue')

    x_axis = plt.gca().xaxis
    x_axis.set_major_locator(mdates.MonthLocator())
    x_axis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    plt.ylabel("Cases")
    plt.legend()
    plt.show()

def main():
    # Load and preprocess data
    data = pd.read_csv("data/Cases.csv").iloc[41:, :]
    data.columns = data.columns.str.strip()
    data[cols := data.columns.drop('date')] = data[cols].apply(pd.to_numeric, errors='coerce')
    data.index = pd.to_datetime(data['date'])
    data.drop(columns=['date'], inplace=True)
    data.fillna(value=0, inplace=True)

    # Smooth data for trend visualization
    smooth_data = data.rolling(7).mean().round(5)

    # Display trends
    show_trends("Daily Vaccination Trend in Penang", 
                smooth_data.index[data.index >= '2021-02-24'], 
                smooth_data['daily'][data.index >= '2021-02-24'])

    show_trends("Daily New Cases in Penang", smooth_data.index, smooth_data['cases_new'])
    show_trends("Daily Active Cases in Penang", smooth_data.index, smooth_data['cases_active'])
    show_trends("Daily New Deaths in Penang", 
                smooth_data.index[data.index >= '2020-03-22'], 
                smooth_data['deaths_new'][data.index >= '2020-03-22'])

    # Plot daily test cases
    plt.title("Daily Test Cases in Penang")
    plt.bar(smooth_data.index[smooth_data['rtk-ag'] > 0], 
            smooth_data['rtk-ag'][smooth_data['rtk-ag'] > 0], 
            label='RTK-AG', color=(0.2, 0.4, 0.65, 0.8))

    plt.bar(smooth_data.index[smooth_data['pcr'] > 0], 
            smooth_data['pcr'][smooth_data['pcr'] > 0], 
            bottom=smooth_data['rtk-ag'][smooth_data['rtk-ag'] > 0], 
            label='PCR', color=(0.2, 0.4, 0.6, 0.6))

    plt.legend()
    x_axis = plt.gca().xaxis
    x_axis.set_major_locator(mdates.MonthLocator())
    x_axis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    plt.ylabel("Cases")
    plt.show()

    # Calculate cumulative deaths
    cumulative_deaths = data['deaths_new'].sum()
    print(f'Cumulative Deaths: {cumulative_deaths}')

    # Calculate vaccine effectiveness
    p, s, az, c = calculate_vaccine_type_for_death()
    total_vaccinated_deaths = p + s + az + c
    print(f'Total Vaccinated Deaths: {total_vaccinated_deaths}')

    total_p = calculate_vaccine_type(data['pfizer1'])
    total_s = calculate_vaccine_type(data['sinovac1'])
    total_az = calculate_vaccine_type(data['astra1'])
    total_c = calculate_vaccine_type(data['cansino'])

    print(total_p, total_s, total_az, total_c)

    p_death_rate = calculate_vax_death_rate(p, total_p)
    s_death_rate = calculate_vax_death_rate(s, total_s)
    az_death_rate = calculate_vax_death_rate(az, total_az)

    print(f"Pfizer Death Rate: {p_death_rate:.3f}%")
    print(f"Sinovac Death Rate: {s_death_rate:.3f}%")
    print(f"AstraZeneca Death Rate: {az_death_rate:.3f}%")

    # Predict deaths and display results
    predict_deaths(data, p_death_rate, s_death_rate, az_death_rate)

if __name__ == "__main__":
    main()
