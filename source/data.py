import pandas as pd

def main():
    # Get the data from GitHub and store it into pandas dataframes  
    vaccine_data = pd.read_csv("https://raw.githubusercontent.com/CITF-Malaysia/citf-public/main/vaccination/vax_state.csv",  
                            usecols=['date', 'state', 'daily', 'cumul', 'cumul_full', 'pfizer1', 'sinovac1', 'astra1', 'sinopharm1', 'cansino'])  
    cases_data = pd.read_csv("https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_state.csv",  
                            usecols=[0, 1, 2, 4, 5, 7, 8])  
    deaths_data = pd.read_csv("https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/deaths_state.csv",  
                            usecols=[0, 1, 2, 6, 7])  
    tests_data = pd.read_csv("https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/tests_state.csv")  
    linelist_deaths = pd.read_csv("https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/linelist/linelist_deaths.csv",  
                                usecols=['state', 'brand1', 'brand2'])  
    
    # Filter only for Pulau Pinang data  
    state_name = "Pulau Pinang"  
    vaccine_data = vaccine_data[vaccine_data['state'] == state_name]  
    cases_data = cases_data[cases_data['state'] == state_name]  
    deaths_data = deaths_data[deaths_data['state'] == state_name]  
    tests_data = tests_data[tests_data['state'] == state_name]  
    linelist_deaths = linelist_deaths[linelist_deaths['state'] == state_name]  
    
    # Convert datetime  
    date_columns = ['date']  
    dataframes = [vaccine_data, cases_data, deaths_data, tests_data]  
    for df in dataframes:  
        df[date_columns] = pd.to_datetime(df[date_columns], format="%Y-%m-%d")  
    
    # Remove the state column  
    dataframes = [vaccine_data, cases_data, deaths_data, tests_data, linelist_deaths]  
    for df in dataframes:  
        del df["state"]  
    
    # Merge all the data into a single file  
    cases_data = pd.merge(cases_data, tests_data, on='date', how='outer')  
    cases_data = pd.merge(cases_data, vaccine_data, on='date', how='outer')  
    cases_data = pd.merge(cases_data, deaths_data, on='date', how='outer')  
    
    # Store into new CSV files  
    cases_data.to_csv("data/Cases.csv", index=False)  
    linelist_deaths.to_csv("data/linelist_deaths.csv", index=False)  

if __name__ == "__main__":
    main()