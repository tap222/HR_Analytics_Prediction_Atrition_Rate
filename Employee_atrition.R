# Load the following packages
library(tidyquant)  # Loads tidyverse and several other pkgs 
library(readxl)     # Super simple excel reader
library(h2o)        # Professional grade ML pkg
library(lime)       # Explain complex black-box ML models

# Read excel data
hr_data_raw <- read_excel(path = "data/WA_Fn-UseC_-HR-Employee-Attrition.xlsx")

# View first 10 rows
hr_data_raw[1:10,] %>%
  knitr::kable(caption = "First 10 rows")

hr_data <- hr_data_raw %>%
  mutate_if(is.character, as.factor) %>%
  select(Attrition, everything())

glimpse(hr_data)

# Initialize H2O JVM
h2o.init()

# Split data into Train/Validation/Test Sets
hr_data_h2o <- as.h2o(hr_data)

split_h2o <- h2o.splitFrame(hr_data_h2o, c(0.7, 0.15), seed = 1234 )

train_h2o <- h2o.assign(split_h2o[[1]], "train" ) # 70%
valid_h2o <- h2o.assign(split_h2o[[2]], "valid" ) # 15%
test_h2o  <- h2o.assign(split_h2o[[3]], "test" )  # 15%

# Set names for h2o
y <- "Attrition"
x <- setdiff(names(train_h2o), y)

# Run the automated machine learning 
automl_models_h2o <- h2o.automl(
  x = x, 
  y = y,
  training_frame    = train_h2o,
  leaderboard_frame = valid_h2o,
  max_runtime_secs  = 30
)

# Extract leader model
automl_leader <- automl_models_h2o@leader

# Predict on hold-out set, test_h2o
pred_h2o <- h2o.predict(object = automl_leader, newdata = test_h2o)

# Prep for performance assessment
test_performance <- test_h2o %>%
  tibble::as_tibble() %>%
  select(Attrition) %>%
  add_column(pred = as.vector(pred_h2o$predict)) %>%
  mutate_if(is.character, as.factor)
test_performance

# Confusion table counts
confusion_matrix <- test_performance %>%
  table() 
confusion_matrix

tn <- confusion_matrix[1]
tp <- confusion_matrix[4]
fp <- confusion_matrix[3]
fn <- confusion_matrix[2]

accuracy <- (tp + tn) / (tp + tn + fp + fn)
misclassification_rate <- 1 - accuracy
recall <- tp / (tp + fn)
precision <- tp / (tp + fp)
null_error_rate <- tn / (tp + tn + fp + fn)

tibble(
  accuracy,
  misclassification_rate,
  recall,
  precision,
  null_error_rate
) %>% 
  transpose() 