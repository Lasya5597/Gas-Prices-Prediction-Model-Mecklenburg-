# Gas-Prices-Prediction-Model-Mecklenburg-
A cost savings by gas station distance index to help users determine whether or not driving to a further gas station for lower prices is actually worth the gas expenditure for driving the extra distance (as opposed to paying a marginally higher price at a closer station) 
DTSC 3601: Data Science Studio 3: Predictive Analysis & Their Impacts

● State which license you will use (1 sentence)

We are going to use the MIT license because it will allow us to have the most amount of permissions, AKA giving us maximum freedom to modify, use, and distribute our code.


● Outline of the classes/functions you expect and how they will work (note this does not need to be complete) (½ page)

We will need to create a cost savings by gas station distance index to help users determine whether or not driving to a further gas station for lower prices is actually worth the gas expenditure for driving the extra distance (as opposed to paying a marginally higher price at a closer station)

GasStation class representing each station, using station_id, brand/company, and location as attributes. This class could include the following functions:
get_price_history() - retrieves historical prices for that station
calculate_avg_price() - calculates its average price over a certain period
calculate_price_change() - measures how much its price has increased/decreased recently
compare_to_area_average() - determines whether the station is normally cheaper or more expensive than nearby stations.

Demand sector class representing demand by residential, commercial, or industrial sector attributes:
sector_id - unique identifier for the specific sector
sector_name - descriptive name of the sector (e.g., "Residential", "Heavy Industrial", "Commercial")
average_consumption_rate - the baseline volume of gas typically consumed by the sector per hour or day
peak_demand_hours - the specific times of day when this sector hits maximum usage (e.g., mornings/evenings for residential)
volatility_index - a measure of how much the sector's demand fluctuates unexpectedly

● What examples you will include for a user to explore your tool (2-3 sentences explaining why you show those examples)

We plan to include an example for a major metro county (e.g., Mecklenburg/Charlotte) showing a typical forecast use case, and a second example for a rural county with sparser data to illustrate how the model handles limited historical records. We'll also include an example comparing predicted vs. actual prices over a past time window to demonstrate model accuracy. These examples were chosen because they show the tool working under both "ideal" (data-rich) and "realistic" (data-sparse) conditions, and because visually validating past predictions helps users trust the forecasts before relying on them for future decisions.
