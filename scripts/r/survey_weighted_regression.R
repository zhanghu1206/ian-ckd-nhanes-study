#!/usr/bin/env Rscript
# ==============================================================================
# IAN-CKD Project: Proper NHANES Survey-Weighted Logistic Regression (V69)
#
# This script performs the design-based survey-weighted analysis described in the
# manuscript (Methods: "Analyses accounted for NHANES' complex survey design
# (WTMEC2YR weights, SDMVPSU, SDMVSTRA)"). It uses the R `survey` package with a
# correct svydesign object (strata = SDMVSTRA, PSU = SDMVPSU, weights = WTMEC2YR,
# nest = TRUE) and svyglm, which yields design-based (not frequency-weight) standard
# errors. This is the gold-standard NHANES variance estimation.
#
# IAN construction replicates scripts/advanced_analysis.py exactly:
#   - tertile cut points derived from the TRAINING cycles (G/H/I, 2011-2016)
#   - NLR scored 0-2 (higher = worse); Hemoglobin & Albumin reverse-coded (2-0)
#   - IAN = NLR_T + HB_T + ALB_T  (range 0-6)
#
# Models:
#   Model 1: CKD ~ IAN
#   Model 2: CKD ~ IAN + age + sex
#   Model 3 (= manuscript "Model 4, survey-weighted"):
#            CKD ~ IAN + age + sex + BMI + diabetes + hypertension
# ==============================================================================

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
  getwd()
})
proj_root <- dirname(dirname(.script_dir))   # scripts/r -> scripts -> proj_root
if (!dir.exists(proj_root)) proj_root <- getwd()

suppressPackageStartupMessages({
  library(survey)
  library(dplyr)
})

message("Project root: ", proj_root)

# ---- Read merged NHANES data (relative path) ----
merged_path <- file.path(proj_root, "output", "nhanes_ckd_merged.csv")
message("Reading: ", merged_path)
df <- read.csv(merged_path, stringsAsFactors = FALSE)

# ---- Analytical sample (matches manuscript n = 20,222) ----
# The manuscript's final analytical sample is adults >=20 with complete
# NLR/Hemoglobin/Albumin (22,617 eligible - 2,395 missing components = 20,222).
# NOTE: RIDEXPRG pregnancy exclusion is mentioned in the text but is NOT a net
# subtraction in the reported arithmetic (22,617 - 2,395 = 20,222). Reproducing
# the reported analytical sample therefore uses complete-case adults >=20 only.
adult <- df %>% filter(RIDAGEYR >= 20)
analytical <- adult %>%
  filter(!is.na(NLR), !is.na(HEMOGLOBIN), !is.na(ALBUMIN), !is.na(CKD))
message("Analytical sample: ", nrow(analytical), " participants")

# ---- Derive tertile cut points from TRAINING cycles (G/H/I) ----
train <- analytical %>% filter(CYCLE %in% c("G", "H", "I"))
qcut_bins <- function(x) {
  q <- quantile(x, probs = c(1/3, 2/3), type = 7, na.rm = TRUE)
  c(-Inf, q[1], q[2], Inf)
}
nlr_b <- qcut_bins(train$NLR)
hb_b  <- qcut_bins(train$HEMOGLOBIN)
alb_b <- qcut_bins(train$ALBUMIN)

assign_ian <- function(d) {
  # NB: as.numeric() on a factor returns the level INDEX (1-3), not the label
  # values; convert via as.character() first to recover the intended 0-2 scores.
  score <- function(x, brk, labs) {
    f <- cut(x, breaks = brk, include.lowest = TRUE, labels = labs)
    as.numeric(as.character(f))
  }
  d$NLR_T <- score(d$NLR,       nlr_b, c(0, 1, 2))
  d$HB_T  <- score(d$HEMOGLOBIN, hb_b,  c(2, 1, 0))
  d$ALB_T <- score(d$ALBUMIN,   alb_b, c(2, 1, 0))
  d$IAN   <- d$NLR_T + d$HB_T + d$ALB_T
  d$SEX   <- factor(ifelse(d$RIAGENDR == 1, "Male", "Female"))
  d$HYPERTENSION <- as.integer(d$BPQ020 == 1)
  d
}
analytical <- assign_ian(analytical)
message("Mean IAN: ", round(mean(analytical$IAN, na.rm = TRUE), 3),
        " (range ", min(analytical$IAN, na.rm = TRUE), "-", max(analytical$IAN, na.rm = TRUE), ")")

# ---- Proper NHANES survey design ----
# nest = TRUE because PSU codes restart within each strata cycle.
design <- svydesign(
  ids    = ~SDMVPSU,
  strata = ~SDMVSTRA,
  weights = ~WTMEC2YR,
  nest   = TRUE,
  data   = analytical,
  survey.lonely.psu = "adjust"
)
message("Survey design created. Number of PSUs: ", length(unique(analytical$SDMVPSU)))

# ---- Fit survey-weighted models ----
fit1 <- svyglm(CKD ~ IAN, family = quasibinomial(), design = design)
fit2 <- svyglm(CKD ~ IAN + RIDAGEYR + SEX, family = quasibinomial(), design = design)
fit3 <- svyglm(CKD ~ IAN + RIDAGEYR + SEX + BMXBMI + DIABETES + HYPERTENSION,
               family = quasibinomial(), design = design)

# ---- Extract OR and design-based 95% CI for IAN ----
extract_ian <- function(fit, model_name) {
  b  <- coef(fit)["IAN"]
  se <- SE(fit)["IAN"]
  or <- exp(b)
  lo <- exp(b - 1.96 * se)
  hi <- exp(b + 1.96 * se)
  data.frame(Model = model_name,
             IAN_OR = round(or, 3),
             CI_low = round(lo, 3),
             CI_high = round(hi, 3),
             stringsAsFactors = FALSE)
}
results <- rbind(
  extract_ian(fit1, "Model 1: IAN only (weighted)"),
  extract_ian(fit2, "Model 2: +age+sex (weighted)"),
  extract_ian(fit3, "Model 3/4: full covariates (weighted, design-based)")
)

# Weighted CKD prevalence for context
prev <- svymean(~CKD, design, na.rm = TRUE)
message("\nWeighted CKD prevalence: ", round(100 * coef(prev)["CKD"], 1), "%")

cat("\n================ Survey-weighted IAN OR (design-based SE) ================\n")
print(results)
cat("===========================================================================\n")

# ---- Save results (relative path) ----
out_path <- file.path(proj_root, "output", "survey_weighted_results.csv")
write.csv(results, out_path, row.names = FALSE)
message("Results written to: ", out_path)
