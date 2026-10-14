import fastf1
import requests
import urllib3
import pandas as pd
import os


os.makedirs('cache', exist_ok=True)

# Enable cache
fastf1.Cache.enable_cache('cache')

# Monkey-patch SSL verification due to Ergast/Jolpica backend issues
urllib3.disable_warnings()
old_req = requests.Session.request
requests.Session.request = lambda *a, **kw: old_req(*a, **{**kw, 'verify': False})

# Enable cache
fastf1.Cache.enable_cache('cache')

def get_event_list(year):
    """Fetches all racing events for a given year."""
    schedule = fastf1.get_event_schedule(year)
    # Filter out testing events, keep only races
    races = schedule[schedule['EventFormat'] != 'testing']
    return races['EventName'].tolist()

def get_qualifying_session(year, event):
    """Loads and returns the Qualifying session object (with telemetry)."""
    session = fastf1.get_session(year, event, 'Q')
    session.load()
    return session

def get_available_drivers_for_q(session, q_part):
    """
    Returns a list of driver abbreviations who set a time in Q1, Q2, or Q3.
    q_part must be 'Q1', 'Q2', or 'Q3'.
    """
    results = session.results
    # Filter drivers who have a valid lap time in the specified Q session
    valid_drivers = results[results[q_part].notna()]
    return valid_drivers['Abbreviation'].tolist()

def fetch_q_lap(session, driver, q_part):
    """
    Finds the exact fastf1 Lap object for a driver that matches their official time in Q1, Q2, or Q3.
    """
    driver_laps = session.laps.pick_driver(driver)
    results = session.results
    
    official_time = results.loc[results['Abbreviation'] == driver, q_part].iloc[0]
    
    # Find the lap that matches the official time within a 2-millisecond tolerance
    matching_laps = driver_laps.loc[(driver_laps['LapTime'] - official_time).dt.total_seconds().abs() < 0.002]
    
    if len(matching_laps) > 0:
        return matching_laps.iloc[0]
    else:
        # Fallback if exact match fails: just return their fastest lap overall
        return driver_laps.pick_fastest()
