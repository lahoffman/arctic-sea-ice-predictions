# arctic-sea-ice-predictions

This file includes scripts used for analysis in the following publication: 
Hoffman, L., Massonnet, F., Sticker, A. (2025). Probabilistic forecasts of interannual September Arctic sea ice extent with data-driven statistical models.

File naming convention:
D: data
M: model
E: evaluation
P: plot
T: table
TS: supplementary table
PS: supplementary plot

Files are dependent on outputs from previous scripts and should be run in convention (D-M-E-P-T-TS-PS) and numerical order. 

File paths (load and save) must be updated according to your appropriate directories. 

Raw data can be downloaded from the following publically-available sources: 
1. Coupled Model Intercomparison Project phase 6 (CMIP6). Historical and SSP5-8.5 simulations. Models: ACCESS-CM2, ACCESS-ESM1-5, CanESM5, IPSL-CM6A, MIROC6, MRO-ESM2-0. Variables: siextentn, siconc, sithick, areacello.
2. National Snow and Ice Data Center (NSIDC) Sa Ice Index. Variable: sea ice extent, northern hemisphere, monthly. 
3. Pan-Arctic Ice Ocean Modeling Assimilation System (PIOMAS). Variable: sea ice thickness, monthly.
