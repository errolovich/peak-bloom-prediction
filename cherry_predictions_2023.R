library(chillR)
library(fable)
library(feasts)
library(keras)
library(readr)
library(tensorflow)
library(tidyverse)
library(tsfgrnn)
library(tsibble)

##### NOTE #####
# Scroll all the way down to the 'Best models' section (right at the bottom) 
# to skip all my data preparation and model selection steps.

##############################################################################

#######################
# Data pre-processing #
#######################
# Import different location data sets
washingtondc <- read_csv()
liestal <- read_csv()
kyoto <- read_csv()


# Function to filter a data set for years 1922-2022
cleaning.data <- function(dataset) {
  work.data <- dataset %>%
    select(location, year, bloom_doy) %>%
    filter(year %in% (1972:2022))
}

# Apply function across data sets with longer historical records
# and combine into single data frame/tibble
older.data <- list(washingtondc, liestal, kyoto)
working.data <- lapply(older.data, cleaning.data) %>% reduce(bind_rows) 

unique(working.data$location) # sanity check

## Dealing with Vancouver
vancouver <- read_csv()
head(vancouver)

# Now add vancouver to working.data data set
working.data <- rbind(working.data, vancouver) %>%
  mutate_at(c('year', 'bloom_doy'), as.integer) # double to integer data types
unique(working.data$location) # sanity check
head(working.data)

# Convert to tsibble
working.data.arima <- as_tsibble(working.data, key = location, index = year)
head(working.data) # sanity check
working.data.arima <- working.data %>% fill_gaps(.full = TRUE)

#######################
# ARIMA               #
#######################

fit <- working.data.arima %>%
  model(arima = ARIMA(bloom_doy ~ pdq(0, 1, 0)))

predictions <- fit %>% forecast(h = 10)
predictions %>% 
  filter(location == 'washingtondc') %>%
  hilo(predictions, level = 95)

#######################
# Time seriese NN.    #
#######################

dc.df <- read_csv('cleaned.dca.weather.csv')
dc.df <- stack_hourly_temps(dc.df, latitude = 38.88)
dc.df <- as.data.frame(do.call(rbind, dc.df))
rownames(dc.df) <- NULL

comp.dc <- read_csv('washingtondc.csv')
comp.dc <- comp.dc %>%
    filter(year > 1972 & year < 2023)

# Function
dc.year <- 1973

dc.accumulated.range <- function(dc.year) {
  doy <- comp.dc$bloom_doy[comp.dc$year == dc.year]
  dc.df %>% filter(Year == dc.year, JDay <= doy)
}


dc.chilling <- function(dc.year) {
  k <- dc.accumulated.range(dc.year)
  chilling(make_JDay(k),Start_JDay = min(k$JDay), End_JDay = max(k$JDay))
}

dc.final <- lapply(1973:2022, dc.chilling) %>% reduce(bind_rows) %>%
  rename(Year = End_year, Bloom_doy = Season_days) %>% select(-c(Season, Data_days))


# Çreate empty rows for 2023 to 2032 for predictions
dc.prediction.df <- dc.final %>% select(-Perc_complete)
tail(dc.prediction.df, 12)

###### CHILLING HOURS
# Additive
ch.additive <- grnn_forecasting(dc.prediction.df$Chilling_Hours, h = 10, transform = 'additive')
ch.additive$prediction
cha.ro <- rolling_origin(ch.additive)
print(cha.ro$global_accu)

ch.multiplicative <- grnn_forecasting(dc.prediction.df$Chilling_Hours, h = 10, transform = 'multiplicative')
ch.multiplicative$prediction
chm.ro <- rolling_origin(ch.multiplicative)
print(chm.ro$global_accu)

# FINAL Chilling_Hours
ch.additive.mimo <- grnn_forecasting(dc.prediction.df$Chilling_Hours, h = 10, msas = 'MIMO', transform = 'additive')
ch.additive.mimo$prediction
cham.ro <- rolling_origin(ch.additive.mimo)
print(cham.ro$global_accu) # MAE: 134.77882

###### UTAH_MODEL
uma.additive <- grnn_forecasting(dc.prediction.df$Utah_Model, h = 10, transform = 'additive')
uma.additive$prediction
uma.ro <- rolling_origin(uma.additive)
print(uma.ro$global_accu) # MAE: 128.70884

umm.multiplicative <- grnn_forecasting(dc.prediction.df$Utah_Model, h = 10, transform = 'multiplicative')
umm.multiplicative$prediction
umm.ro <- rolling_origin(umm.multiplicative)
print(umm.ro$global_accu) # MAE: 130.64593

# FINAL Utah_Model
uma.additive.mimo <- grnn_forecasting(dc.prediction.df$Utah_Model, h = 10, msas = 'MIMO', transform = 'additive')
uma.additive.mimo$prediction
umam.ro <- rolling_origin(uma.additive.mimo)
print(umam.ro$global_accu) # MAE: 116.67317

###### CHILL_PORTIONS
cp.additive.mimo <- grnn_forecasting(dc.prediction.df$Chill_portions, h = 10, msas = 'MIMO', transform = 'additive')
cp.additive.mimo$prediction
cpa.ro <- rolling_origin(cp.additive.mimo)
print(cpa.ro$global_accu) # MAE: 4.512405

###### GDH
gdh.additive.mimo <- grnn_forecasting(dc.prediction.df$GDH, h = 10, msas = 'MIMO', transform = 'additive')
gdh.additive.mimo$prediction
gdha.ro <- rolling_origin(gdh.additive.mimo)
print(gdha.ro$global_accu) # MAE: 1262.00429

gdh.additive <- grnn_forecasting(dc.prediction.df$GDH, h = 10, transform = 'additive')
gdh.additive$prediction
gdh.ro <- rolling_origin(gdh.additive)
print(gdh.ro$global_accu) # MAE: 1184.36290

# FINAL GDH
gdh.multiplicative <- grnn_forecasting(dc.prediction.df$GDH, h = 10, transform = 'multiplicative')
gdh.multiplicative$prediction
gdhm.ro <- rolling_origin(gdh.multiplicative)
print(gdhm.ro$global_accu) # MAE: 1170.76451


dc.nn.pred <- data.frame(Chilling_Hours = ch.additive.mimo$prediction,
           Utah_Model = uma.additive.mimo$prediction,
           Chill_portions = cp.additive.mimo$prediction,
           GDH = gdh.multiplicative$prediction)
nn.pred.matrix <- as.matrix(dc.nn.pred)
dimnames(nn.pred.matrix) <- NULL

nn.model1 <- load_model_hdf5('dc.model.h5')

dc.nn.fc <- nn.model1 %>% predict(nn.pred.matrix)
dc.nn.fc


#### Additive / MIMO
# 1973-2022
BD1 <- grnn_forecasting(dc.prediction.df$Bloom_doy, h = 10, msas = 'MIMO', transform = 'additive')
BD1$prediction
BD1.ro <- rolling_origin(BD1)
print(BD1.ro$global_accu) # MAE: 6.738990

# 1921-2022
BD2 <- grnn_forecasting(washingtondc$bloom_doy, h = 10, msas = 'MIMO', transform = 'additive')
BD2$prediction
BD2.ro <- rolling_origin(BD2)
print(BD2.ro$global_accu) # MAE: 6.667098

# #### Additive / no MIMO
# 1973-2022
BD3 <- grnn_forecasting(dc.prediction.df$Bloom_doy, h = 10, transform = 'additive')
BD3$prediction
BD3.ro <- rolling_origin(BD3)
print(BD3.ro$global_accu) # MAE: 6.705867

# 1921-2022
BD4 <- grnn_forecasting(washingtondc$bloom_doy, h = 10, transform = 'additive')
BD4$prediction
BD4.ro <- rolling_origin(BD4)
print(BD4.ro$global_accu) # MAE: 7.718608

#### Multiplicative / no MIMO
# 1973-2022
BD5 <- grnn_forecasting(dc.prediction.df$Bloom_doy, h = 10, msas = 'MIMO', transform = 'multiplicative')
BD5$prediction
BD5.ro <- rolling_origin(BD5)
print(BD5.ro$global_accu) # MAE: 6.743584

# 1921-2022
BD6 <- grnn_forecasting(washingtondc$bloom_doy, h = 10, msas = 'MIMO', transform = 'multiplicative')
BD6$prediction
BD6.ro <- rolling_origin(BD6)
print(BD6.ro$global_accu) # MAE: 6.837825

#### Multiplicative / no MIMO
# 1973-2022
BD7 <- grnn_forecasting(dc.prediction.df$Bloom_doy, h = 10, transform = 'multiplicative')
BD7$prediction
BD7.ro <- rolling_origin(BD7)
print(BD7.ro$global_accu) # MAE: 6.617282

# 1921-2022
BD8 <- grnn_forecasting(washingtondc$bloom_doy, h = 10, transform = 'multiplicative')
BD8$prediction
BD8.ro <- rolling_origin(BD8)
print(BD8.ro$global_accu) # MAE: 7.818455

#### Additive / MIMO; no transformation
# 1973-2022
BD9 <- grnn_forecasting(dc.prediction.df$Bloom_doy, h = 10, msas = 'MIMO', transform = 'none')
BD9$prediction
BD9.ro <- rolling_origin(BD9)
print(BD9.ro$global_accu) # MAE: 6.738990

#### Additive / no MIMO; no transformation
# 1973-2022
BD10 <- grnn_forecasting(dc.prediction.df$Bloom_doy, h = 10, transform = 'none')
BD10$prediction
BD10.ro <- rolling_origin(BD10)
print(BD10.ro$global_accu) # MAE: 6.553783

#######################
# DC predictions.     # --------> BD10
#######################

#### Additive / no MIMO; no transformation
# 1973-2022
BD10 <- grnn_forecasting(dc.prediction.df$Bloom_doy, h = 10, transform = 'none')
BD10$prediction
BD10.ro <- rolling_origin(BD10)
print(BD10.ro$global_accu) # MAE: 6.553783

#########################
# Vancouver predictions # --------> vanc.BD4
#########################
head(vanc.prediction.df)

#### Additive / no MIMO
# 1999-2022
vanc.BD1 <- grnn_forecasting(vanc.prediction.df$Bloom_doy, h = 10, transform = 'additive')
vanc.BD1$prediction
vanc.BD1.ro <- rolling_origin(vanc.BD1)
print(vanc.BD1.ro$global_accu) # MAE: 10.93771

#### Multiplicative / no MIMO
# 1999-2022
vanc.BD2 <- grnn_forecasting(vanc.prediction.df$Bloom_doy, h = 10, transform = 'multiplicative')
vanc.BD2$prediction
vanc.BD2.ro <- rolling_origin(vanc.BD2)
print(vanc.BD2.ro$global_accu) # MAE: 10.44942

#### Additive / no MIMO; no transformation: vanc.prediction.df
# 1999-2022
vanc.BD3 <- grnn_forecasting(vanc.prediction.df$Bloom_doy, h = 10, transform = 'none')
vanc.BD3$prediction
vanc.BD3.ro <- rolling_origin(vanc.BD3)
print(vanc.BD3.ro$global_accu) # MAE: 6.608878

#### No MIMO; no transformation: vancouver df
# 1999-2022
vanc.BD4 <- grnn_forecasting(vancouver$bloom_doy, h = 10, transform = 'none')
vanc.BD4$prediction
vanc.BD4.ro <- rolling_origin(vanc.BD4)
print(vanc.BD4.ro$global_accu) # MAE: 5.690125

#########################
# Liestal predictions   # --------> liestal.BD3
#########################

liestal.trimmed <- liestal %>%
  filter(year >= 1973 & year < 2023)

#### No MIMO; no transformation
liestal.BD1 <- grnn_forecasting(liestal.trimmed$bloom_doy, h = 10, transform = 'none')
liestal.BD1$prediction
liestal.trimmed.ro1 <- rolling_origin(liestal.BD1)
print(liestal.trimmed.ro1$global_accu) # MAE: 10.11080

#### Additive / No MIMO
liestal.BD2 <- grnn_forecasting(liestal.trimmed$bloom_doy, h = 10, transform = 'additive')
liestal.BD2$prediction
liestal.trimmed.ro2 <- rolling_origin(liestal.BD2)
print(liestal.trimmed.ro2$global_accu) # MAE: 8.555194

#### Additive / MIMO
liestal.BD3 <- grnn_forecasting(liestal.trimmed$bloom_doy, h = 10, msas = 'MIMO', transform = 'additive')
liestal.BD3$prediction
liestal.trimmed.ro3 <- rolling_origin(liestal.BD3)
print(liestal.trimmed.ro3$global_accu) # MAE: 8.094408

#### Multiplicative / MIMO
liestal.BD4 <- grnn_forecasting(liestal.trimmed$bloom_doy, h = 10, msas = 'MIMO', transform = 'multiplicative')
liestal.BD4$prediction
liestal.trimmed.ro4 <- rolling_origin(liestal.BD4)
print(liestal.trimmed.ro4$global_accu) # MAE: 8.269290

#### Multiplicative / no MIMO
liestal.BD5 <- grnn_forecasting(liestal.trimmed$bloom_doy, h = 10, transform = 'multiplicative')
liestal.BD5$prediction
liestal.trimmed.ro5 <- rolling_origin(liestal.BD5)
print(liestal.trimmed.ro5$global_accu) # MAE: 8.803972

#########################
# Kyoto predictions     # --------> kyoto.BD3
#########################

kyoto.trimmed <- kyoto %>%
  filter(year >= 1973 & year < 2023)

#### No MIMO; no transformation
kyoto.BD1 <- grnn_forecasting(kyoto.trimmed$bloom_doy, h = 10, transform = 'none')
kyoto.BD1$prediction
kyoto.trimmed.ro1 <- rolling_origin(kyoto.BD1)
print(kyoto.trimmed.ro1$global_accu) # MAE: 5.814376

#### None / MIMO
kyoto.BD2 <- grnn_forecasting(kyoto.trimmed$bloom_doy, h = 10, msas = 'MIMO', transform = 'none')
kyoto.BD2$prediction
kyoto.trimmed.ro2 <- rolling_origin(kyoto.BD2)
print(kyoto.trimmed.ro2$global_accu) # MAE: 5.309744

#### Additive / MIMO
kyoto.BD3 <- grnn_forecasting(kyoto.trimmed$bloom_doy, h = 10, msas = 'MIMO', transform = 'additive')
kyoto.BD3$prediction
kyoto.trimmed.ro3 <- rolling_origin(kyoto.BD3)
print(kyoto.trimmed.ro3$global_accu) # MAE: 4.141640

#### Multiplicative / MIMO
kyoto.BD4 <- grnn_forecasting(kyoto.trimmed$bloom_doy, h = 10, msas = 'MIMO', transform = 'multiplicative')
kyoto.BD4$prediction
kyoto.trimmed.ro4 <- rolling_origin(kyoto.BD4)
print(kyoto.trimmed.ro4$global_accu) # MAE: 4.172814

#### Multiplicative / no MIMO
kyoto.BD5 <- grnn_forecasting(kyoto.trimmed$bloom_doy, h = 10, transform = 'multiplicative')
kyoto.BD5$prediction
kyoto.trimmed.ro5 <- rolling_origin(kyoto.BD5)
print(kyoto.trimmed.ro5$global_accu) # MAE: 4.512922

##############################################################################

#########################
# Best models           # 
#########################

# Kyoto
kyoto.BD3 <- grnn_forecasting(kyoto.trimmed$bloom_doy, h = 10, msas = 'MIMO', transform = 'additive')
kyoto.BD3$prediction
kyoto.trimmed.ro3 <- rolling_origin(kyoto.BD3)
print(kyoto.trimmed.ro3$global_accu) # MAE: 4.141640

# Liestal
liestal.BD3 <- grnn_forecasting(liestal.trimmed$bloom_doy, h = 10, msas = 'MIMO', transform = 'additive')
liestal.BD3$prediction
liestal.trimmed.ro3 <- rolling_origin(liestal.BD3)
print(liestal.trimmed.ro3$global_accu) # MAE: 8.094408

# DC
BD10 <- grnn_forecasting(dc.prediction.df$Bloom_doy, h = 10, transform = 'none')
BD10$prediction
BD10.ro <- rolling_origin(BD10)
print(BD10.ro$global_accu) # MAE: 6.553783

# Vancouver
vanc.BD4 <- grnn_forecasting(vancouver$bloom_doy, h = 10, transform = 'none')
vanc.BD4$prediction
vanc.BD4.ro <- rolling_origin(vanc.BD4)
print(vanc.BD4.ro$global_accu) # MAE: 5.690125

#########################
# Output csv            # 
#########################

final.df <- data.frame(year = 2023:2032,
                       kyoto = kyoto.BD3$prediction,
                       liestal = liestal.BD3$prediction,
                       washingtondc = BD10$prediction,
                       vancouver = vanc.BD4$prediction) %>% round(digits = 0)
final.df

# Save final.df as csv
write_csv(final.df, 'cherry_blossoms_2023_predictions.csv')
