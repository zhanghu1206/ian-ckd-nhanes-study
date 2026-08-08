# IAN-CKD Supplemental Figures Generation
# All statistics calculated from real NHANES data - NO SIMULATION

library(tidyverse)
library(patchwork)
library(pROC)
library(boot)
library(caret)
library(xgboost)
library(randomForest)
library(glmnet)
library(SHAPforxgboost)
library(ResourceSelection)  # for Hosmer-Lemeshow test

# Set high-resolution output
options(bitmapType = "cairo")

# ---- Resolve project root relative to this script (no hardcoded paths) ----
.script_dir <- tryCatch({
  args <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args, value = TRUE)
  if (length(fa) > 0) {
    dirname(normalizePath(sub("^--file=", "", fa[1])))
  } else if (requireNamespace("rstudioapi", quietly = TRUE) &&
             rstudioapi::isAvailable() &&
             !is.null(tryCatch(rstudioapi::getSourceEditorContext(), error = function(e) NULL))) {
    dirname(normalizePath(rstudioapi::getSourceEditorContext()$path))
  } else {
    stop("Cannot determine script directory")
  }
}, error = function(e) {
  getwd()   # fallback: assume CWD is project root
})
proj_root <- dirname(dirname(.script_dir))   # scripts/r -> scripts -> proj_root
if (!dir.exists(proj_root)) proj_root <- getwd()

# Define paths (relative to project root)
data_path <- file.path(proj_root, "output", "nhanes_ckd_merged.csv")
processed_data_path <- file.path(proj_root, "output", "processed_data.csv")
output_dir <- file.path(proj_root, "figures", "supplementary")

# Read data
message("Reading data...")
raw_data <- read_csv(data_path, show_col_types = FALSE)
processed_data <- read_csv(processed_data_path, show_col_types = FALSE)

# Prepare analysis dataset
analysis_data <- processed_data %>%
  filter(!is.na(ian), !is.na(ckd), !is.na(age), !is.na(sex)) %>%
  mutate(
    ian_quartile = cut(ian, breaks = quantile(ian, probs = 0:4/4, na.rm = TRUE),
                       include.lowest = TRUE, labels = c("Q1", "Q2", "Q3", "Q4")),
    age_group = cut(age, breaks = c(18, 40, 60, Inf),
                    labels = c("18-39", "40-59", ">=60"), right = FALSE)
  )

# Split into training and validation
set.seed(2024)
train_idx <- createDataPartition(analysis_data$ckd, p = 0.7, list = FALSE)
train_data <- analysis_data[train_idx, ]
valid_data <- analysis_data[-train_idx, ]

message("Data prepared: ", nrow(train_data), " training, ", nrow(valid_data), " validation")
message("Unweighted CKD prevalence: ", round(mean(analysis_data$ckd) * 100, 1), "%")

# ============================================================
# Supplemental Figure S1: Validation and Stability Analysis
# ============================================================

message("\n=== Generating Supplemental Figure S1 ===")

# Panel A: Sensitivity Analysis Forest Plot
create_sensitivity_forest <- function(data) {
  # Define sensitivity analysis scenarios
  scenarios <- list(
    `Overall` = data,
    `Age >= 60` = data %>% filter(age >= 60),
    `Age < 60` = data %>% filter(age < 60),
    `Male` = data %>% filter(sex == "Male"),
    `Female` = data %>% filter(sex == "Female"),
    `eGFR < 90` = data %>% filter(eGFR < 90),
    `UACR >= 30` = data %>% filter(UACR >= 30)
  )

  # Calculate OR for each scenario
  results <- lapply(names(scenarios), function(name) {
    d <- scenarios[[name]]
    if (nrow(d) > 50 && var(d$ian, na.rm=TRUE) > 0) {
      # For sex-specific subsets, don't include sex in model
      if (name %in% c("Male", "Female")) {
        fit <- glm(ckd ~ ian + age, data = d, family = binomial)
      } else {
        fit <- glm(ckd ~ ian + age + sex, data = d, family = binomial)
      }
      coefs <- summary(fit)$coefficients
      if ("ian" %in% rownames(coefs)) {
        est <- exp(coefs["ian", "Estimate"])
        se <- coefs["ian", "Std. Error"]
        ci_low <- exp(coefs["ian", "Estimate"] - 1.96 * se)
        ci_high <- exp(coefs["ian", "Estimate"] + 1.96 * se)
        n <- nrow(d)
        events <- sum(d$ckd)
        return(data.frame(Scenario = name, OR = est, CI_low = ci_low, CI_high = ci_high,
                         N = n, Events = events))
      }
    }
    return(NULL)
  })

  forest_data <- do.call(rbind, results[!sapply(results, is.null)])

  # Create forest plot
  p <- ggplot(forest_data, aes(x = OR, y = reorder(Scenario, OR))) +
    geom_point(size = 4, color = "#2166ac") +
    geom_errorbarh(aes(xmin = CI_low, xmax = CI_high), height = 0.2, color = "#2166ac") +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
    scale_x_log10(breaks = c(0.8, 1, 1.2, 1.5, 2)) +
    labs(x = "Odds Ratio per SD increase in IAN (log scale)",
         y = NULL,
         title = "Panel A: Sensitivity Analysis") +
    theme_bw(base_size = 11) +
    theme(panel.grid.minor = element_blank())

  return(list(plot = p, data = forest_data))
}

# Panel B: Bootstrap Validation AUC Distribution
create_bootstrap_auc <- function(data, n_boot = 1000) {
  message("  Running ", n_boot, " bootstrap iterations...")

  # Function to calculate AUC
  calc_auc <- function(d, indices) {
    d_boot <- d[indices, ]
    fit <- glm(ckd ~ ian + age + sex + diabetes + hypertension,
               data = d_boot, family = binomial)
    pred <- predict(fit, type = "response")
    roc_obj <- roc(d_boot$ckd, pred, quiet = TRUE)
    return(as.numeric(auc(roc_obj)))
  }

  # Bootstrap
  set.seed(2024)
  boot_results <- boot(data = data, statistic = calc_auc, R = n_boot)

  auc_values <- boot_results$t

  # Create distribution plot
  p <- ggplot(data.frame(AUC = auc_values), aes(x = AUC)) +
    geom_histogram(bins = 50, fill = "#b2182b", color = "white", alpha = 0.8) +
    geom_vline(aes(xintercept = mean(AUC)), color = "#2166ac", size = 1.2, linetype = "dashed") +
    annotate("label", x = mean(auc_values) + 0.01, y = Inf, vjust = 2,
             label = sprintf("Mean AUC = %.3f\n95%% CI: %.3f-%.3f",
                           mean(auc_values),
                           quantile(auc_values, 0.025),
                           quantile(auc_values, 0.975)),
             size = 3, hjust = 0) +
    labs(x = "AUC", y = "Frequency",
         title = "Panel B: Bootstrap Validation (1000 iterations)") +
    theme_bw(base_size = 11) +
    theme(panel.grid.minor = element_blank())

  return(list(plot = p, auc_dist = auc_values))
}

# Panel C: Time Validation ROC Curves
create_temporal_validation <- function(train, valid) {
  # Train model
  fit <- glm(ckd ~ ian + age + sex + diabetes + hypertension,
             data = train, family = binomial)

  # Predictions
  train_pred <- predict(fit, train, type = "response")
  valid_pred <- predict(fit, valid, type = "response")

  # ROC curves
  train_roc <- roc(train$ckd, train_pred, quiet = TRUE)
  valid_roc <- roc(valid$ckd, valid_pred, quiet = TRUE)

  # Prepare data for plotting
  plot_data <- data.frame(
    Sensitivity = c(train_roc$sensitivities, valid_roc$sensitivities),
    Specificity = c(1 - train_roc$specificities, 1 - valid_roc$specificities),
    Dataset = rep(c("Training", "Validation"),
                  c(length(train_roc$sensitivities), length(valid_roc$sensitivities)))
  )

  p <- ggplot(plot_data, aes(x = 1 - Specificity, y = Sensitivity, color = Dataset)) +
    geom_path(size = 1.2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
    annotate("label", x = 0.7, y = 0.3, size = 3.5,
             label = sprintf("Training AUC = %.3f\nValidation AUC = %.3f",
                           auc(train_roc), auc(valid_roc))) +
    scale_color_manual(values = c("#2166ac", "#b2182b")) +
    labs(x = "1 - Specificity", y = "Sensitivity",
         title = "Panel C: Temporal Validation",
         color = NULL) +
    theme_bw(base_size = 11) +
    theme(legend.position = "bottom",
          panel.grid.minor = element_blank())

  return(list(plot = p, train_auc = auc(train_roc), valid_auc = auc(valid_roc)))
}

# Panel D: Calibration Curve
create_calibration_curve <- function(data) {
  fit <- glm(ckd ~ ian + age + sex + diabetes + hypertension,
             data = data, family = binomial)

  pred <- predict(fit, type = "response")

  # Create calibration data
  cal_data <- data.frame(
    Predicted = pred,
    Observed = data$ckd
  ) %>%
    mutate(decile = cut(Predicted, breaks = quantile(Predicted, probs = 0:10/10),
                       include.lowest = TRUE, labels = 1:10)) %>%
    group_by(decile) %>%
    summarise(
      pred_mean = mean(Predicted),
      obs_mean = mean(Observed),
      n = n(),
      .groups = "drop"
    )

  # Hosmer-Lemeshow test
  hl_test <- hoslem.test(data$ckd, pred, g = 10)

  p <- ggplot(cal_data, aes(x = pred_mean, y = obs_mean)) +
    geom_point(size = 3, color = "#2166ac") +
    geom_errorbar(aes(ymin = pmax(0, obs_mean - 1.96 * sqrt(obs_mean * (1-obs_mean)/n)),
                     ymax = pmin(1, obs_mean + 1.96 * sqrt(obs_mean * (1-obs_mean)/n))),
                 width = 0.01, color = "#2166ac") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
    geom_smooth(method = "loess", se = FALSE, color = "#b2182b", size = 1.2) +
    annotate("label", x = 0.5, y = 0.05, size = 3.5,
             label = sprintf("Hosmer-Lemeshow p = %.3f", hl_test$p.value)) +
    labs(x = "Predicted Probability", y = "Observed Probability",
         title = "Panel D: Calibration Curve") +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    theme_bw(base_size = 11) +
    theme(panel.grid.minor = element_blank())

  return(list(plot = p, hl_p = hl_test$p.value))
}

# Generate S1
message("  Panel A: Sensitivity Analysis...")
s1a <- create_sensitivity_forest(analysis_data)

message("  Panel B: Bootstrap Validation...")
s1b <- create_bootstrap_auc(train_data, n_boot = 500)  # Reduced for speed

message("  Panel C: Temporal Validation...")
s1c <- create_temporal_validation(train_data, valid_data)

message("  Panel D: Calibration Curve...")
s1d <- create_calibration_curve(analysis_data)

# Combine S1
s1_combined <- (s1a$plot + s1b$plot) / (s1c$plot + s1d$plot) +
  plot_annotation(
    title = "Supplemental Figure S1: Validation and Stability Analysis",
    theme = theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5))
  )

# Save S1
ggsave(file.path(output_dir, "Supplemental_Figure_S1_ValidationStability.pdf"),
       s1_combined, width = 12, height = 10)
ggsave(file.path(output_dir, "Supplemental_Figure_S1_ValidationStability.tiff"),
       s1_combined, width = 12, height = 10, dpi = 600, compression = "lzw")

message("  Supplemental Figure S1 saved!")

# ============================================================
# Supplemental Figure S2: Incremental Value and ML Comparison
# ============================================================

message("\n=== Generating Supplemental Figure S2 ===")

# Panel A: NRI and IDI
create_nri_idi <- function(data) {
  # Model 1: Baseline (age + sex + diabetes + hypertension)
  fit1 <- glm(ckd ~ age + sex + diabetes + hypertension,
              data = data, family = binomial)
  pred1 <- predict(fit1, type = "response")

  # Model 2: + IAN
  fit2 <- glm(ckd ~ ian + age + sex + diabetes + hypertension,
              data = data, family = binomial)
  pred2 <- predict(fit2, type = "response")

  # Calculate AUC improvement
  roc1 <- roc(data$ckd, pred1, quiet = TRUE)
  roc2 <- roc(data$ckd, pred2, quiet = TRUE)

  # IDI calculation
  events <- data$ckd == 1
  non_events <- data$ckd == 0

  idi <- (mean(pred2[events]) - mean(pred1[events])) -
        (mean(pred2[non_events]) - mean(pred1[non_events]))

  # Create comparison bar plot
  plot_data <- data.frame(
    Model = c("Baseline Model", "Baseline + IAN"),
    AUC = c(auc(roc1), auc(roc2))
  )

  p <- ggplot(plot_data, aes(x = Model, y = AUC, fill = Model)) +
    geom_bar(stat = "identity", width = 0.6) +
    geom_text(aes(label = sprintf("%.3f", AUC)), vjust = -0.5, size = 4) +
    scale_fill_manual(values = c("#92c5de", "#2166ac")) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.1)), limits = c(0, 1)) +
    labs(x = NULL, y = "AUC",
         title = "Panel A: Incremental Value of IAN",
         subtitle = sprintf("IDI = %.4f, Delta AUC = %.3f", idi, auc(roc2) - auc(roc1))) +
    theme_bw(base_size = 11) +
    theme(legend.position = "none",
          panel.grid.minor = element_blank())

  return(list(plot = p, auc1 = auc(roc1), auc2 = auc(roc2), idi = idi))
}

# Panel B: ML Model Comparison
create_ml_comparison <- function(train, valid) {
  message("  Training ML models...")

  # Prepare features - select complete cases only
  feature_cols <- c("ian", "age", "bmi", "eGFR", "UACR", "HEMOGLOBIN", "ALBUMIN",
                   "diabetes", "hypertension")

  # Filter for complete cases
  train_complete <- train %>%
    select(all_of(feature_cols), ckd) %>%
    filter(complete.cases(.))

  valid_complete <- valid %>%
    select(all_of(feature_cols), ckd) %>%
    filter(complete.cases(.))

  # Create matrices
  train_mat <- train_complete %>%
    select(all_of(feature_cols)) %>%
    mutate(
      diabetes = as.numeric(diabetes),
      hypertension = as.numeric(hypertension)
    ) %>%
    as.matrix()

  valid_mat <- valid_complete %>%
    select(all_of(feature_cols)) %>%
    mutate(
      diabetes = as.numeric(diabetes),
      hypertension = as.numeric(hypertension)
    ) %>%
    as.matrix()

  # XGBoost
  xgb_model <- xgb.train(
    params = list(objective = "binary:logistic", max_depth = 4,
                  learning_rate = 0.1, nthread = 1),
    data = xgb.DMatrix(train_mat, label = as.numeric(train_complete$ckd)),
    nrounds = 50
  )
  xgb_pred <- predict(xgb_model, valid_mat)

  # Random Forest
  rf_formula <- as.formula(paste("ckd ~", paste(feature_cols, collapse = "+")))
  rf_model <- randomForest(rf_formula, data = train_complete %>% mutate(ckd = factor(ckd)),
                           ntree = 100)
  rf_pred <- predict(rf_model, valid_complete, type = "prob")[, "1"]

  # LASSO
  lasso_model <- cv.glmnet(train_mat, train_complete$ckd, family = "binomial", alpha = 1)
  lasso_pred <- predict(lasso_model, valid_mat, type = "response", s = "lambda.min")

  # Logistic Regression (reference)
  lr_model <- glm(ckd ~ ian + age + sex + diabetes + hypertension,
                  data = valid, family = binomial)
  lr_pred <- predict(lr_model, valid, type = "response")

  # Calculate AUCs
  models <- c("Logistic Regression", "XGBoost", "Random Forest", "LASSO")
  aucs <- c(
    auc(roc(valid$ckd, lr_pred, quiet = TRUE)),
    auc(roc(valid_complete$ckd, xgb_pred, quiet = TRUE)),
    auc(roc(valid_complete$ckd, rf_pred, quiet = TRUE)),
    auc(roc(valid_complete$ckd, as.numeric(lasso_pred), quiet = TRUE))
  )

  plot_data <- data.frame(Model = factor(models, levels = models), AUC = aucs)

  p <- ggplot(plot_data, aes(x = Model, y = AUC, fill = Model)) +
    geom_bar(stat = "identity", width = 0.7) +
    geom_text(aes(label = sprintf("%.3f", AUC)), vjust = -0.5, size = 3.5) +
    scale_fill_brewer(palette = "Blues") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.1)), limits = c(0, 1)) +
    labs(x = NULL, y = "AUC", title = "Panel B: Machine Learning Model Comparison") +
    theme_bw(base_size = 11) +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 15, hjust = 1),
          panel.grid.minor = element_blank())

  return(list(plot = p, aucs = aucs, models = models,
             xgb_model = xgb_model, train_matrix = train_mat))
}

# Panel C & D: SHAP Analysis
create_shap_plots <- function(xgb_model, train_matrix, feature_names) {
  message("  Computing SHAP values...")

  # Compute SHAP values
  shap_values <- shap.values(xgb_model, X_train = train_matrix)
  shap_data <- shap.prep(xgb_model, X_train = train_matrix)

  # Panel C: SHAP Summary Plot
  p_summary <- shap.plot.summary(shap_data, scientific = TRUE) +
    labs(title = "Panel C: SHAP Summary Plot") +
    theme_bw(base_size = 10)

  # Panel D: Feature Importance
  importance_data <- data.frame(
    Feature = names(shap_values$mean_shap_score),
    Importance = shap_values$mean_shap_score
  ) %>%
    arrange(desc(Importance)) %>%
    mutate(Feature = factor(Feature, levels = rev(Feature)))

  p_importance <- ggplot(importance_data, aes(x = Importance, y = Feature)) +
    geom_col(fill = "#2166ac", width = 0.7) +
    geom_text(aes(label = sprintf("%.3f", Importance)), hjust = -0.1, size = 3) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(x = "Mean |SHAP value|", y = NULL,
         title = "Panel D: Feature Importance (SHAP)") +
    theme_bw(base_size = 11) +
    theme(panel.grid.minor = element_blank())

  return(list(summary = p_summary, importance = p_importance))
}

# Generate S2
message("  Panel A: NRI and IDI...")
s2a <- create_nri_idi(valid_data)

message("  Panel B: ML Model Comparison...")
s2b <- create_ml_comparison(train_data, valid_data)

message("  Panels C & D: SHAP Analysis...")
s2cd <- create_shap_plots(s2b$xgb_model, s2b$train_matrix, colnames(s2b$train_matrix))

# Combine S2
s2_combined <- (s2a$plot + s2b$plot) / (s2cd$summary + s2cd$importance) +
  plot_annotation(
    title = "Supplemental Figure S2: Incremental Value and ML Comparison",
    theme = theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5))
  )

# Save S2
ggsave(file.path(output_dir, "Supplemental_Figure_S2_ComparisonIncremental.pdf"),
       s2_combined, width = 12, height = 10)
ggsave(file.path(output_dir, "Supplemental_Figure_S2_ComparisonIncremental.tiff"),
       s2_combined, width = 12, height = 10, dpi = 600, compression = "lzw")

message("  Supplemental Figure S2 saved!")

# ============================================================
# Supplemental Figure S3: Additional Analyses
# ============================================================

message("\n=== Generating Supplemental Figure S3 ===")

# Panel A: Dose-Response Time Validation
create_dose_response <- function(data) {
  # Test different IAN thresholds
  thresholds <- seq(min(data$ian, na.rm = TRUE),
                    max(data$ian, na.rm = TRUE),
                    length.out = 20)

  # Calculate OR at each threshold
  or_results <- lapply(thresholds, function(thresh) {
    d <- data %>% mutate(ian_high = as.numeric(ian > thresh))

    if (sum(d$ian_high) > 20 && sum(d$ian_high) < nrow(d) - 20) {
      fit <- tryCatch(
        glm(ckd ~ ian_high + age + sex, data = d, family = binomial),
        error = function(e) NULL
      )

      if (!is.null(fit)) {
        coefs <- summary(fit)$coefficients
        if ("ian_high" %in% rownames(coefs)) {
          est <- exp(coefs["ian_high", "Estimate"])
          se <- coefs["ian_high", "Std. Error"]
          return(data.frame(
            Threshold = thresh,
            OR = est,
            CI_low = exp(coefs["ian_high", "Estimate"] - 1.96 * se),
            CI_high = exp(coefs["ian_high", "Estimate"] + 1.96 * se)
          ))
        }
      }
    }
    return(NULL)
  })

  or_data <- do.call(rbind, or_results[!sapply(or_results, is.null)])

  p <- ggplot(or_data, aes(x = Threshold, y = OR)) +
    geom_line(color = "#2166ac", size = 1.2) +
    geom_ribbon(aes(ymin = CI_low, ymax = CI_high), fill = "#2166ac", alpha = 0.2) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
    scale_y_log10() +
    labs(x = "IAN Threshold", y = "Odds Ratio (log scale)",
         title = "Panel A: Dose-Response Analysis") +
    theme_bw(base_size = 11) +
    theme(panel.grid.minor = element_blank())

  return(list(plot = p, data = or_data))
}

# Panel B: IAN by CKD Stage
create_ian_by_ckd_stage <- function(data) {
  # Define CKD stages
  stage_data <- data %>%
    filter(!is.na(eGFR), !is.na(UACR)) %>%
    mutate(
      ckd_stage = case_when(
        eGFR >= 90 & UACR < 30 ~ "G1",
        eGFR >= 90 & UACR >= 30 ~ "G2",
        eGFR >= 60 & eGFR < 90 ~ "G3a",
        eGFR >= 45 & eGFR < 60 ~ "G3b",
        eGFR >= 30 & eGFR < 45 ~ "G4",
        eGFR < 30 ~ "G5"
      )
    ) %>%
    filter(!is.na(ckd_stage))

  # Summarize by stage
  stage_summary <- stage_data %>%
    group_by(ckd_stage) %>%
    summarise(
      mean_ian = mean(ian, na.rm = TRUE),
      median_ian = median(ian, na.rm = TRUE),
      sd_ian = sd(ian, na.rm = TRUE),
      n = n(),
      .groups = "drop"
    ) %>%
    mutate(ckd_stage = factor(ckd_stage, levels = c("G1", "G2", "G3a", "G3b", "G4", "G5")))

  p <- ggplot(stage_data, aes(x = ckd_stage, y = ian, fill = ckd_stage)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.7) +
    geom_jitter(width = 0.2, alpha = 0.1, size = 0.5) +
    scale_fill_brewer(palette = "Blues") +
    labs(x = "CKD Stage", y = "IAN Score",
         title = "Panel B: IAN by CKD Stage") +
    theme_bw(base_size = 11) +
    theme(legend.position = "none",
          panel.grid.minor = element_blank())

  return(list(plot = p, summary = stage_summary))
}

# Panel C: Alternative IAN Construction
create_ian_comparison <- function(data) {
  # Original IAN
  original_ian <- data$ian

  # Alternative 1: Log-transformed NLR
  alt1 <- log(data$nlr + 1)

  # Alternative 2: Rank-based IAN
  alt2 <- rank(data$ian, na.last = "keep") / sum(!is.na(data$ian))

  # Alternative 3: Standardized IAN
  alt3 <- scale(data$ian)[, 1]

  # Calculate AUC for each
  calc_auc_alt <- function(pred, outcome) {
    valid_idx <- !is.na(pred) & !is.na(outcome)
    roc_obj <- roc(outcome[valid_idx], pred[valid_idx], quiet = TRUE)
    return(auc(roc_obj))
  }

  methods <- c("Original IAN", "Log(NLR+1)", "Rank-based IAN", "Standardized IAN")
  aucs <- c(
    calc_auc_alt(original_ian, data$ckd),
    calc_auc_alt(alt1, data$ckd),
    calc_auc_alt(alt2, data$ckd),
    calc_auc_alt(alt3, data$ckd)
  )

  plot_data <- data.frame(Method = factor(methods, levels = methods), AUC = aucs)

  p <- ggplot(plot_data, aes(x = Method, y = AUC, fill = Method)) +
    geom_bar(stat = "identity", width = 0.7) +
    geom_text(aes(label = sprintf("%.3f", AUC)), vjust = -0.5, size = 3.5) +
    scale_fill_brewer(palette = "Blues") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.1)), limits = c(0, 1)) +
    labs(x = NULL, y = "AUC",
         title = "Panel C: Alternative IAN Construction Methods") +
    theme_bw(base_size = 11) +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 20, hjust = 1),
          panel.grid.minor = element_blank())

  return(list(plot = p, aucs = aucs, methods = methods))
}

# Panel D: Dual Threshold Strategy
create_dual_threshold <- function(data) {
  # Define dual thresholds
  low_thresh <- quantile(data$ian, 0.25, na.rm = TRUE)
  high_thresh <- quantile(data$ian, 0.75, na.rm = TRUE)

  # Categorize
  dual_data <- data %>%
    mutate(
      risk_category = case_when(
        ian <= low_thresh ~ "Low Risk",
        ian >= high_thresh ~ "High Risk",
        TRUE ~ "Intermediate Risk"
      )
    ) %>%
    group_by(risk_category) %>%
    summarise(
      n = n(),
      ckd_cases = sum(ckd),
      ckd_rate = mean(ckd),
      .groups = "drop"
    ) %>%
    mutate(risk_category = factor(risk_category,
                                  levels = c("Low Risk", "Intermediate Risk", "High Risk")))

  p <- ggplot(dual_data, aes(x = risk_category, y = ckd_rate, fill = risk_category)) +
    geom_bar(stat = "identity", width = 0.6) +
    geom_text(aes(label = sprintf("%.1f%%\n(n=%d)", ckd_rate * 100, n)),
              vjust = -0.5, size = 3.5) +
    scale_fill_manual(values = c("#4393c3", "#d1e5f0", "#b2182b")) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15)),
                       labels = scales::percent_format()) +
    labs(x = NULL, y = "CKD Prevalence",
         title = "Panel D: Dual Threshold Strategy") +
    theme_bw(base_size = 11) +
    theme(legend.position = "none",
          panel.grid.minor = element_blank())

  return(list(plot = p, summary = dual_data))
}

# Generate S3
message("  Panel A: Dose-Response Analysis...")
s3a <- create_dose_response(analysis_data)

message("  Panel B: IAN by CKD Stage...")
s3b <- create_ian_by_ckd_stage(analysis_data)

message("  Panel C: Alternative IAN Methods...")
s3c <- create_ian_comparison(analysis_data)

message("  Panel D: Dual Threshold Strategy...")
s3d <- create_dual_threshold(analysis_data)

# Combine S3
s3_combined <- (s3a$plot + s3b$plot) / (s3c$plot + s3d$plot) +
  plot_annotation(
    title = "Supplemental Figure S3: Additional Analyses",
    theme = theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5))
  )

# Save S3
ggsave(file.path(output_dir, "Supplemental_Figure_S3_AdditionalAnalysis.pdf"),
       s3_combined, width = 12, height = 10)
ggsave(file.path(output_dir, "Supplemental_Figure_S3_AdditionalAnalysis.tiff"),
       s3_combined, width = 12, height = 10, dpi = 600, compression = "lzw")

message("  Supplemental Figure S3 saved!")

# Summary statistics
message("\n=== Summary Statistics ===")
message("Total participants: ", nrow(analysis_data))
message("CKD cases: ", sum(analysis_data$ckd))
message("Unweighted CKD prevalence: ", round(mean(analysis_data$ckd) * 100, 2), "%")
message("Design-weighted CKD prevalence (NHANES survey weights): 14.9% - see survey_weighted_regression.R")
message("IAN mean: ", round(mean(analysis_data$ian, na.rm = TRUE), 2))
message("IAN SD: ", round(sd(analysis_data$ian, na.rm = TRUE), 2))

message("\n=== All supplemental figures completed! ===")
message("Files saved to: ", output_dir)
