# SIADS 696 - Milestone II: Uber Driving/Riding :  Predicting Outcomes and Discovering Patterns

# Notebooks requirement

This Project run stable on Python version 3.9 and 3.9.16, any later version might cause unknown incompatibility.

Users should install all the required packages using the `requirements.txt`. This can be done by using the PIP command.

```
pip install -r requirements.txt
```

NOTE: If you can't install the virtual environment, please feel free to reach out to us

# Rerunning data collection step

We have already cached the datasets into the `datasets/raw` directory through csv files. But in case users want to collect the data again, Notebook #1 contains helper functions that allow user to generate the data. If you're interested in manually collecting a certain dataset, delete the related csv file in the `datasets/raw` and run the data collection notebook again. It should pick up and collect the data. Please note that secondary datasets will require an API key, which you might need to generate yourself.

# Rerunning data processing step

In the `datasets/raw/merged` directory, we have saved a cleaned version of the merged and scaled dataset that we are using for the modeling, it's called `ncr_ride_bookings_with_weather_filled_scaled_short.csv`. If you want to rerun the whole preprocessing step, you can run notebook 2.1-2.3 in the given order.

# Modeling step

We've done 2 versions of model training, one is a lite version, which we used the top 20-30 features that correlate with the target feature, this reduce the training time significantly. We did this because we expected we might not have enough time to run hyperparameter optimization using the whole dataset. Later into the final weeks of the project, we set up Great Lake HPC and had successfuly train on the whole dataset. That's where there is a `_lite` version for both supervised and unsupervised learning part.

# Contributors
* Kha Nguyen (minhkha@umich.edu)
* Naiwen Duan (nduan@umich.edu)
* Ching-Yao Lin (yaulin@umich.edu)