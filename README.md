# Predicting ICU Mortality

This project is using the MIMIC-III database (https://mimic.physionet.org/) to predict ICU mortality.

I've loaded in the data on a PSQL server, following the instructions available on the MIMIC database website. Starting from there, the work in this repo does the following:

1. Identifies how many measurements per itemid (measurement type) there are, ie. how many temperature measurements were taken.
2. Creates the PSQL code to merge split itemids, and convert units for itemids with mismatched units (ie. converting all the Farenheight temperatures to Celcius).
3. For a list of patients and itemids, grabs the most recent measurements at 24 hours after admission. This is used as the feature/target table. 
4. Based on the feature/target table, we kick everything into XGBoost. We look at the train/test accuracy, and take a quick look at the feature importances. 
