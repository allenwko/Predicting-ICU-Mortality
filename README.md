# Predicting ICU Mortality

This project is using the MIMIC-III database (https://mimic.physionet.org/) to predict ICU mortality.

There are three main things I want to do:

1. Make a classifier model to predict if someone will survive their ICU stay, based on information available 24 hours after admission. 
2. In order to improve #1, I want to apply a clustering algorithim on patients at various times throughout their ICU stay to see how patients' measurement "move" through their stay.
3. Build an app to serve this model on the internet. 
