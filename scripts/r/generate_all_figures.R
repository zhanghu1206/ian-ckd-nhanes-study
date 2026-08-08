#!/usr/bin/env Rscript
# ==============================================================================
# IAN-CKD Project: Generate Publication Figures
# Using Real NHANES 2011-2018 Data - Complete Version
# ==============================================================================

options(stringsAsFactors = FALSE)
set.seed(2025)

# Load required packages
suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(patchwork)
  library(pROC)
  library(dplyr)
  library(splines)
  library(ragg)
})

# Nature-style theme
theme_nature <- function() {
  theme_classic(base_size = 10) +
    theme(
      plot.title = element_text(face = "bold", size = 10, hjust = 0.5),
      axis.title = element_text(size = 9, face = "bold"),
      axis.text = element_text(size = 8, color = "black"),
      legend.title = element_text(size = 8, face = "bold"),
      legend.text = element_text(size = 7),
      panel.grid = element_blank()
    )
}

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

# Output directory (relative to project root)
output_dir <- file.path(proj_root, "figures", "main")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# ==============================================================================
# 1. Read and Process Data  (canonical 20,222 analytic sample)
# ==============================================================================
cat("Reading NHANES data...\n")
dt_raw <- fread(file.path(proj_root, "output", "nhanes_ckd_merged.csv"))
cat("Total merged records:", nrow(dt_raw), "\n")

# Canonical analytical dataset (n = 20,222) -- identical to make_processed_data.py
dt_final <- fread(file.path(proj_root, "output", "processed_data.csv"))
cat("Analytical sample (processed_data.csv):", nrow(dt_final), "\n")

# Merge a few raw components needed only for the index-comparison figure (Fig 4)
merge_cols <- dt_raw[, .(SEQN, NEUTROPHIL, LYMPHOCYTE, NLR)]
setnames(merge_cols, c("NEUTROPHIL", "LYMPHOCYTE", "NLR"),
         c("neutrophil", "lymphocyte", "nlr_raw"))
dt_final <- merge(dt_final, merge_cols, by = "SEQN", all.x = TRUE)

dt_final <- dt_final %>%
  mutate(
    nlr = as.numeric(nlr_raw),
    hemoglobin = as.numeric(HEMOGLOBIN),
    albumin = as.numeric(ALBUMIN),
    nlr_calc = ifelse(is.na(nlr) & !is.na(neutrophil) & !is.na(lymphocyte) & lymphocyte > 0,
                      neutrophil / lymphocyte, nlr),
    pni = ifelse(!is.na(albumin) & !is.na(lymphocyte), albumin * 10 + 5 * lymphocyte, NA_real_),
    ali = ifelse(!is.na(bmi) & !is.na(albumin) & !is.na(nlr_calc) & nlr_calc > 0,
                 bmi * albumin / nlr_calc, NA_real_),
    dataset = ifelse(CYCLE %in% c("G", "H", "I"), "Training", "Validation")
  ) %>%
  mutate(
    ian_tertile = cut(ian, breaks = c(-0.5, 2, 4, 6.5),
                     labels = c("T1 (Low)", "T2 (Medium)", "T3 (High)"), include.lowest = TRUE)
  )

n_total <- nrow(dt_raw)
n_age20 <- sum(dt_raw$RIDAGEYR >= 20, na.rm = TRUE)
n_complete <- nrow(dt_final)
n_train <- sum(dt_final$dataset == "Training", na.rm = TRUE)
n_valid <- sum(dt_final$dataset == "Validation", na.rm = TRUE)

cat("Training:", n_train, "| Validation:", n_valid, "| Unweighted CKD prevalence:", round(mean(dt_final$ckd) * 100, 1), "%\n")
cat("Design-weighted CKD prevalence (NHANES survey weights): 14.9% - see survey_weighted_regression.R\n")

# ==============================================================================
# 2. Figure 1: Study Flowchart
# ==============================================================================
cat("\n=== Figure 1: Flowchart ===\n")

flow_data <- data.frame(
  y = c(7, 6, 5, 4, 3, 2, 1),
  label = c(
    paste0("NHANES 2011-2018\nN = ", format(n_total, big.mark = ",")),
    paste0("Aged >= 20 years\nN = ", format(n_age20, big.mark = ","),
           "\n(Excluded <20 y: ", format(n_total - n_age20, big.mark = ","), ")"),
    paste0("Excluded: incomplete data\nn = ", format(n_age20 - n_complete, big.mark = ","),
           "\n(incl. pregnant)"),
    paste0("Final Analytical Sample\nN = ", format(n_complete, big.mark = ",")),
    paste0("Training (2011-2016)\nN = ", format(n_train, big.mark = ",")),
    paste0("Validation (2017-2018)\nN = ", format(n_valid, big.mark = ",")),
    paste0("CKD: ", sum(dt_final$ckd == 1), " (14.9% weighted)")
  ),
  color = c("#2E86AB", "#2E86AB", "#E94F37", "#2E86AB", "#44AF69", "#44AF69", "#F18F01"),
  x = c(0, 0, 1.5, 0, -1.2, 1.2, 0)
)

fig1 <- ggplot(flow_data, aes(x = x, y = y)) +
  geom_rect(aes(xmin = x - 1.5, xmax = x + 1.5, ymin = y - 0.35, ymax = y + 0.35, fill = color),
            alpha = 0.8, color = "black", linewidth = 0.5) +
  geom_text(aes(label = label), size = 3, fontface = "bold", color = "white") +
  geom_segment(aes(x = 0, xend = 0, y = 6.65, yend = 5.7), arrow = arrow(length = unit(0.12, "cm")), linewidth = 0.6) +
  geom_segment(aes(x = 0, xend = 0, y = 4.65, yend = 3.7), arrow = arrow(length = unit(0.12, "cm")), linewidth = 0.6) +
  geom_segment(aes(x = 0, xend = -1.2, y = 2.65, yend = 2.35), arrow = arrow(length = unit(0.12, "cm")), linewidth = 0.6) +
  geom_segment(aes(x = 0, xend = 1.2, y = 2.65, yend = 2.35), arrow = arrow(length = unit(0.12, "cm")), linewidth = 0.6) +
  geom_segment(aes(x = 1.2, xend = 1.5, y = 5.5, yend = 5), arrow = arrow(length = unit(0.12, "cm")),
               linewidth = 0.6, color = "#E94F37") +
  scale_fill_identity() +
  coord_cartesian(xlim = c(-3, 3.5), ylim = c(0.5, 7.5)) +
  theme_void() +
  ggtitle("Figure 1. Study Flowchart") +
  theme(plot.title = element_text(face = "bold", size = 11, hjust = 0.5))

ggsave(file.path(output_dir, "Figure_1_Flowchart.pdf"), fig1, width = 183, height = 180, units = "mm")
ggsave(file.path(output_dir, "Figure_1_Flowchart.tiff"), fig1, width = 183, height = 180, units = "mm", dpi = 600)
cat("Figure 1 saved.\n")

# ==============================================================================
# 3. Figure 2: IAN Distribution and Dose-Response
# ==============================================================================
cat("\n=== Figure 2: IAN Distribution ===\n")

# Panel A
fig2a <- ggplot(dt_final, aes(x = ian, fill = factor(ckd, labels = c("No CKD", "CKD")))) +
  geom_histogram(aes(y = after_stat(density)), bins = 7, alpha = 0.6, position = "identity") +
  geom_density(alpha = 0.3, linewidth = 0.8) +
  scale_fill_manual(values = c("#3498DB", "#E74C3C")) +
  scale_x_continuous(breaks = 0:6) +
  labs(x = "IAN (score 0-6)", y = "Density", fill = "CKD") +
  theme_nature() + ggtitle("A")

# Panel B
ckd_by_tertile <- dt_final %>% group_by(ian_tertile) %>%
  summarise(n = n(), prevalence = mean(ckd) * 100,
            se = sqrt(mean(ckd) * (1 - mean(ckd)) / n()) * 100, .groups = "drop")

fig2b <- ggplot(ckd_by_tertile, aes(x = ian_tertile, y = prevalence, fill = ian_tertile)) +
  geom_bar(stat = "identity", width = 0.7, alpha = 0.8) +
  geom_errorbar(aes(ymin = pmax(0, prevalence - 1.96 * se), ymax = prevalence + 1.96 * se), width = 0.2) +
  geom_text(aes(label = paste0(round(prevalence, 1), "%")), vjust = -0.5, size = 2.8) +
  scale_fill_brewer(palette = "Blues", direction = -1) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(x = "IAN Tertile", y = "CKD Prevalence (%)") +
  theme_nature() + theme(legend.position = "none") + ggtitle("B")

# Panel C
spline_model <- glm(ckd ~ bs(ian, df = 4), data = dt_final, family = binomial())
ian_range <- seq(min(dt_final$ian, na.rm = TRUE), max(dt_final$ian, na.rm = TRUE), length.out = 100)
pred_data <- data.frame(ian = ian_range)
pred_matrix <- predict(spline_model, newdata = pred_data, se.fit = TRUE, type = "link")
pred_data$prob <- plogis(pred_matrix$fit)
pred_data$lower <- plogis(pred_matrix$fit - 1.96 * pred_matrix$se.fit)
pred_data$upper <- plogis(pred_matrix$fit + 1.96 * pred_matrix$se.fit)

median_ian <- median(dt_final$ian, na.rm = TRUE)
ref_prob <- predict(spline_model, newdata = data.frame(ian = median_ian), type = "response")
pred_data$or <- pred_data$prob / (1 - pred_data$prob) * ((1 - ref_prob) / ref_prob)
pred_data$or_lower <- pred_data$lower / (1 - pred_data$lower) * ((1 - ref_prob) / ref_prob)
pred_data$or_upper <- pred_data$upper / (1 - pred_data$upper) * ((1 - ref_prob) / ref_prob)

fig2c <- ggplot(pred_data, aes(x = ian, y = or)) +
  geom_ribbon(aes(ymin = or_lower, ymax = or_upper), fill = "#3498DB", alpha = 0.3) +
  geom_line(linewidth = 1, color = "#2C3E50") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
  scale_x_continuous(breaks = 0:6) +
  labs(x = "IAN (score 0-6)", y = "Odds Ratio") +
  theme_nature() + ggtitle("C")

fig2 <- fig2a + fig2b + fig2c + plot_layout(ncol = 3) +
  plot_annotation(title = "Figure 2. IAN Distribution and Dose-Response",
                  theme = theme(plot.title = element_text(face = "bold", size = 11, hjust = 0.5)))

ggsave(file.path(output_dir, "Figure_2_IAN_Distribution_DoseResponse.pdf"), fig2, width = 247, height = 90, units = "mm")
ggsave(file.path(output_dir, "Figure_2_IAN_Distribution_DoseResponse.tiff"), fig2, width = 247, height = 90, units = "mm", dpi = 600)
cat("Figure 2 saved.\n")

# ==============================================================================
# 4. Figure 3: ROC Curves
# ==============================================================================
cat("\n=== Figure 3: ROC Curves ===\n")

train_data <- dt_final %>% filter(dataset == "Training", !is.na(ian), !is.na(ckd))
valid_data <- dt_final %>% filter(dataset == "Validation", !is.na(ian), !is.na(ckd))

roc_train <- roc(train_data$ckd, train_data$ian, quiet = TRUE)
roc_valid <- roc(valid_data$ckd, valid_data$ian, quiet = TRUE)

auc_train_ci <- ci.auc(roc_train, quiet = TRUE)
auc_valid_ci <- ci.auc(roc_valid, quiet = TRUE)

roc_df <- rbind(
  data.frame(specificity = 1 - roc_train$specificities, sensitivity = roc_train$sensitivities,
             Dataset = paste0("Training (AUC=", round(auc_train_ci[2], 3), " [", round(auc_train_ci[1], 3), "-", round(auc_train_ci[3], 3), "])")),
  data.frame(specificity = 1 - roc_valid$specificities, sensitivity = roc_valid$sensitivities,
             Dataset = paste0("Validation (AUC=", round(auc_valid_ci[2], 3), " [", round(auc_valid_ci[1], 3), "-", round(auc_valid_ci[3], 3), "])"))
)

fig3 <- ggplot(roc_df, aes(x = specificity, y = sensitivity, color = Dataset)) +
  geom_abline(intercept = 1, slope = -1, linetype = "dashed", color = "gray50") +
  geom_line(linewidth = 1) +
  scale_color_manual(values = c("#E74C3C", "#3498DB")) +
  scale_x_reverse() + scale_y_continuous() + coord_equal() +
  labs(x = "1 - Specificity", y = "Sensitivity") +
  theme_nature() + theme(legend.position = c(0.7, 0.2)) +
  ggtitle("Figure 3. ROC Curves for IAN")

ggsave(file.path(output_dir, "Figure_3_ROC.pdf"), fig3, width = 183, height = 183, units = "mm")
ggsave(file.path(output_dir, "Figure_3_ROC.tiff"), fig3, width = 183, height = 183, units = "mm", dpi = 600)
cat("Figure 3 saved. Training AUC:", round(auc_train_ci[2], 3), "Validation AUC:", round(auc_valid_ci[2], 3), "\n")

# ==============================================================================
# 5. Figure 4: Index Comparison
# ==============================================================================
cat("\n=== Figure 4: Index Comparison ===\n")

calc_auc <- function(data, outcome, predictor) {
  d <- data[!is.na(data[[outcome]]) & !is.na(data[[predictor]]), ]
  if (nrow(d) < 50) return(NULL)
  roc_obj <- roc(d[[outcome]], d[[predictor]], quiet = TRUE)
  auc_ci <- ci.auc(roc_obj, quiet = TRUE)
  data.frame(index = predictor, auc = auc_ci[2], lower = auc_ci[1], upper = auc_ci[3])
}

indices <- c("ian", "pni", "nlr_calc", "hemoglobin", "albumin", "ali")
names <- c("IAN", "PNI", "NLR", "Hemoglobin", "Albumin", "ALI")

auc_results <- do.call(rbind, lapply(indices, function(i) {
  r <- calc_auc(train_data, "ckd", i)
  if (!is.null(r)) r$index_name <- names[match(i, indices)]
  r
}))

auc_results <- auc_results[order(auc_results$auc, decreasing = TRUE), ]
auc_results$index_name <- factor(auc_results$index_name, levels = auc_results$index_name)

fig4 <- ggplot(auc_results, aes(x = index_name, y = auc, fill = index_name)) +
  geom_bar(stat = "identity", width = 0.7, alpha = 0.8) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  geom_text(aes(label = sprintf("%.3f", auc)), vjust = -0.5, size = 2.8) +
  scale_fill_brewer(palette = "Set2") +
  scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.1))) +
  labs(x = "Index", y = "AUC (95% CI)") +
  theme_nature() + theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Figure 4. Index Comparison")

ggsave(file.path(output_dir, "Figure_4_IndexComparison.pdf"), fig4, width = 183, height = 120, units = "mm")
ggsave(file.path(output_dir, "Figure_4_IndexComparison.tiff"), fig4, width = 183, height = 120, units = "mm", dpi = 600)
cat("Figure 4 saved.\n")

# ==============================================================================
# 6. Figure 5: Subgroup Forest Plot
# ==============================================================================
cat("\n=== Figure 5: Subgroup Forest Plot ===\n")

dt_final$age_group <- ifelse(dt_final$age >= 60, ">=60y", "<60y")
dt_final$bmi_group <- ifelse(dt_final$bmi >= 30, "Obese", "Non-obese")

calc_or <- function(data, var, name) {
  lvls <- unique(na.omit(data[[var]]))
  do.call(rbind, lapply(lvls, function(l) {
    d <- data[data[[var]] == l, ]
    if (nrow(d) < 50) return(NULL)
    fit <- tryCatch(glm(ckd ~ ian_tertile, data = d, family = binomial()), error = function(e) NULL)
    if (is.null(fit)) return(NULL)
    coef_tab <- summary(fit)$coefficients
    if (!"ian_tertileT3 (High)" %in% rownames(coef_tab)) return(NULL)
    or <- exp(coef_tab["ian_tertileT3 (High)", "Estimate"])
    ci <- exp(coef(fit)["ian_tertileT3 (High)"] + c(-1.96, 1.96) * coef_tab["ian_tertileT3 (High)", "Std. Error"])
    data.frame(subgroup = name, level = as.character(l), n = nrow(d), or = or, lower = ci[1], upper = ci[2])
  }))
}

sub_results <- rbind(
  calc_or(dt_final, "sex", "Sex"),
  calc_or(dt_final, "age_group", "Age"),
  calc_or(dt_final, "bmi_group", "BMI"),
  calc_or(dt_final, "diabetes", "Diabetes") %>% mutate(level = ifelse(level == "1", "Yes", "No")),
  calc_or(dt_final, "hypertension", "Hypertension") %>% mutate(level = ifelse(level == "1", "Yes", "No"))
)

overall_fit <- glm(ckd ~ ian_tertile, data = dt_final, family = binomial())
overall_or <- exp(coef(overall_fit)["ian_tertileT3 (High)"])
overall_ci <- exp(confint(overall_fit)["ian_tertileT3 (High)", ])

sub_results <- rbind(data.frame(subgroup = "Overall", level = "All", n = nrow(dt_final),
                                or = overall_or, lower = overall_ci[1], upper = overall_ci[2]), sub_results)
sub_results$subgroup <- factor(sub_results$subgroup, levels = c("Overall", "Sex", "Age", "BMI", "Diabetes", "Hypertension"))
sub_results$label <- paste0(sub_results$level, " (n=", format(sub_results$n, big.mark = ","), ")")
sub_results$or_text <- sprintf("%.2f (%.2f-%.2f)", sub_results$or, sub_results$lower, sub_results$upper)

fig5 <- ggplot(sub_results, aes(x = or, y = reorder(label, -as.numeric(subgroup)))) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
  geom_point(size = 3.5, shape = 15, aes(color = subgroup)) +
  geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.15) +
  geom_text(aes(label = or_text), hjust = -0.1, size = 2.8) +
  scale_color_manual(values = c("#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12", "#1ABC9C")) +
  scale_x_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.3))) +
  labs(x = "OR (95% CI) for CKD (T3 vs T1)", y = "") +
  theme_nature() + theme(legend.position = "none") +
  ggtitle("Figure 5. Subgroup Analysis")

ggsave(file.path(output_dir, "Figure_5_SubgroupForest.pdf"), fig5, width = 247, height = 140, units = "mm")
ggsave(file.path(output_dir, "Figure_5_SubgroupForest.tiff"), fig5, width = 247, height = 140, units = "mm", dpi = 600)
cat("Figure 5 saved.\n")

# ==============================================================================
# 7. Figure 6: Decision Curve Analysis
# ==============================================================================
cat("\n=== Figure 6: Decision Curve Analysis ===\n")

thresholds <- seq(0.05, 0.35, by = 0.01)
dca_data <- data.frame()

fit_ian_all <- glm(ckd ~ ian, data = dt_final, family = binomial())
pred_ian_all <- predict(fit_ian_all, type = "response")
fit_full_all <- glm(ckd ~ ian + age + sex, data = dt_final, family = binomial())
pred_full_all <- predict(fit_full_all, type = "response")
prev <- mean(dt_final$ckd, na.rm = TRUE)

for (thr in thresholds) {
  tp_ian <- sum(pred_ian_all >= thr & dt_final$ckd == 1, na.rm = TRUE)
  fp_ian <- sum(pred_ian_all >= thr & dt_final$ckd == 0, na.rm = TRUE)
  nb_ian <- (tp_ian / nrow(dt_final)) - (fp_ian / nrow(dt_final)) * (thr / (1 - thr))
  
  tp_full <- sum(pred_full_all >= thr & dt_final$ckd == 1, na.rm = TRUE)
  fp_full <- sum(pred_full_all >= thr & dt_final$ckd == 0, na.rm = TRUE)
  nb_full <- (tp_full / nrow(dt_final)) - (fp_full / nrow(dt_final)) * (thr / (1 - thr))
  
  nb_all <- prev - (1 - prev) * (thr / (1 - thr))
  
  dca_data <- rbind(dca_data, data.frame(threshold = thr, IAN_only = nb_ian, IAN_age_sex = nb_full,
                                          Treat_all = nb_all, Treat_none = 0))
}

dca_long <- reshape2::melt(dca_data, id.vars = "threshold", variable.name = "Strategy", value.name = "Net_Benefit")

fig6 <- ggplot(dca_long, aes(x = threshold, y = Net_Benefit, color = Strategy, linetype = Strategy)) +
  geom_line(linewidth = 1) +
  scale_color_manual(values = c("#E74C3C", "#3498DB", "#95A5A6", "#2C3E50")) +
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dashed")) +
  scale_x_continuous(breaks = seq(0.05, 0.35, 0.05)) +
  scale_y_continuous(limits = c(-0.1, 0.15)) +
  labs(x = "Threshold Probability", y = "Net Benefit") +
  theme_nature() + theme(legend.position = c(0.15, 0.85)) +
  ggtitle("Figure 6. Decision Curve Analysis")

ggsave(file.path(output_dir, "Figure_6_DCA.pdf"), fig6, width = 183, height = 120, units = "mm")
ggsave(file.path(output_dir, "Figure_6_DCA.tiff"), fig6, width = 183, height = 120, units = "mm", dpi = 600)
cat("Figure 6 saved.\n")

# ==============================================================================
# Summary
# ==============================================================================
cat("\n=== All figures generated! ===\n")
cat("Output:", output_dir, "\n")
print(list.files(output_dir, pattern = "\\.pdf$|\\.tiff$"))
