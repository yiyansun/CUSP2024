import requests
import json
from datetime import datetime
import os

def get_sites_info(min_date, max_date):
    url = "https://api.erg.ic.ac.uk/AirQuality/Information/MonitoringSiteSpecies/GroupName=London/Json"
    res = requests.request("get", url)
    data = res.json()
    print("Got sites info.")
    for site in data["Sites"]["Site"]:
        mind = datetime.strptime(min_date, "%Y-%m-%d")
        maxd = datetime.strptime(max_date, "%Y-%m-%d")
        site_species = site["Species"]
        if isinstance(site_species, dict):
            site_species = [site_species, ]
        for species in site_species:
            start = datetime.strptime(species["@DateMeasurementStarted"], "%Y-%m-%d %H:%M:%S")
            if start > maxd:
                continue
            if not species["@DateMeasurementFinished"]:
                end = maxd
            else:
                end = datetime.strptime(species["@DateMeasurementFinished"], "%Y-%m-%d %H:%M:%S")
            if end < mind:
                continue


            start_used = max(start, mind).strftime("%Y-%m-%d")
            end_used = min(end, maxd).strftime("%Y-%m-%d")
            save_site_species_data(site["@SiteCode"], start_used, end_used, species["@SpeciesCode"])


def save_site_species_data(sitecode, start, end, species) -> dict:
    url = "https://api.erg.ic.ac.uk/AirQuality/Data/SiteSpecies/SiteCode={}/SpeciesCode={}/StartDate={}/EndDate={}/Json".format(sitecode, species, start, end)
    res = requests.request("get", url)
    print("Saving info fot sitecode={}, species={}, start={}, end={}".format(sitecode, species, start, end))
    __save_site_species_dict(res.json(), start, end)

def get_site_species_timeframe(sitecode, species, start, end):
    filename = "{}_{}_{}_{}.json".format(sitecode, species, start, end)
    cwd = os.getcwd()
    path = os.path.join(cwd, "data")
    path = os.path.join(path, filename)
    file = open(path, "r")
    data = json.loads(file.read())
    ret = {}
    for measurement in data["RawAQData"]["Data"]:
        date = measurement["@MeasurementDateGMT"]
        date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        if measurement["@Value"]:
            ret[date] = float(measurement["@Value"])
    return ret

def __save_site_species_dict(data, start, end):
    file = open(data["RawAQData"]["@SiteCode"] + "_" + data["RawAQData"]["@SpeciesCode"] + "_" + start + "_" + end + ".json", "w")
    json.dump(data, file)
    file.close()

start_date = "2023-10-27"
end_date = "2024-01-15"


#get_sites_info(start_date, end_date)
#save_site_species_data("CE2", start_date, end_date, "NO2")
print(get_site_species_timeframe("BQ7", "PM25", start_date, end_date))