#!/usr/bin/env Rscript
# =============================================================================
# IAN-CKD two-sample Mendelian randomization (TwoSampleMR / MR-PRESSO) -- robust version
#
# Exposures: the four biological components of IAN (UKB, European ancestry)
#   Albumin    ebi-a-GCST90025992  (n~400,938)
#   Hemoglobin ebi-a-GCST90025969  (n~445,373)
#   Neutrophil ebi-a-GCST90025977
#   Lymphocyte ebi-a-GCST90025984
# Outcomes: non-UKB binary CKD (no sample overlap)
#   CKD_CKDGen  ieu-a-1102              (primary analysis)
#   CKD_FinnGen finn-b-N14_CHRONKIDNEYDIS (sensitivity)
#
# Robustness: each (exposure x outcome) pair is wrapped in tryCatch, so a
#   failure in one pair does not stop the whole run; for binary outcomes,
#   get_r_from_lor(..., prevalence=) is used to pre-compute r.outcome for
#   better Steiger directionality precision (prevalence derived from
#   ncase/ncontrol).
# =============================================================================
suppressMessages({
  library(TwoSampleMR)
  library(ieugwasr)
  library(dplyr); library(ggplot2)
})
has_presso <- suppressWarnings(suppressMessages(require(MRPRESSO, quietly=TRUE)))
if(!has_presso) cat("NOTE: MRPRESSO not installed; MR-PRESSO analysis will be skipped\n")
options(width=160, scipen=10)

args <- commandArgs(trailingOnly=TRUE)
OUT  <- ifelse(length(args)>=1, args[1], ".")
dir.create(OUT, showWarnings=FALSE, recursive=TRUE)
cat("== MR start ==\n"); print(Sys.time())
cat("OPENGWAS_JWT loaded:", nchar(Sys.getenv("OPENGWAS_JWT"))>0, "\n")

exposures <- c(
  Albumin    = "ebi-a-GCST90025992",
  Hemoglobin = "ebi-a-GCST90025969",
  Neutrophil = "ebi-a-GCST90025977",
  Lymphocyte = "ebi-a-GCST90025984"
)
outcomes <- list(
  CKD_CKDGen  = list(id = "ieu-a-1102",                binary = TRUE),
  CKD_FinnGen = list(id = "finn-b-N14_CHRONKIDNEYDIS", binary = TRUE)
)

# Startup validation: print sample sizes and traits for all IDs (also checks API connectivity)
cat("\n--- Exposure GWAS info ---\n")
exp_info <- tryCatch(gwasinfo(exposures), error=function(e){cat("  [WARN] gwasinfo exposures failed:", conditionMessage(e),"\n"); NULL})
if(!is.null(exp_info)) print(exp_info)
cat("\n--- Outcome GWAS info ---\n")
out_ids <- sapply(outcomes, function(x) x$id)
out_info <- tryCatch(gwasinfo(out_ids), error=function(e){cat("  [WARN] gwasinfo outcomes failed:", conditionMessage(e),"\n"); NULL})
if(!is.null(out_info)) print(out_info)

mr_all     <- data.frame()
hetero_all <- data.frame()
egger_all  <- data.frame()
steiger_all<- data.frame()
presso_all <- data.frame()
pair_err   <- c()   # record failed pairs

MR_METHODS <- c("mr_ivw","mr_egger_regression","mr_weighted_median","mr_wald_ratio",
                "mr_simple_mode","mr_weighted_mode")

for(nm in names(exposures)){
  eid <- exposures[[nm]]
  cat(sprintf("\n========== Extract exposure instruments: %s (%s) ==========\n", nm, eid))
  exp <- tryCatch(extract_instruments(eid, p1=5e-8, clump=TRUE, r2=0.001, kb=10000),
                  error=function(e){ cat("  [ERR] extract_instruments:", conditionMessage(e),"\n"); NULL })
  if(is.null(exp) || nrow(exp)==0){ cat("  [SKIP] no available instruments\n"); next }
  cat(sprintf("  instruments (after clump): %d\n", nrow(exp)))

  for(onm in names(outcomes)){
    o <- outcomes[[onm]]; oid <- o$id
    cat(sprintf("\n----- MR: %s -> %s (%s) -----\n", nm, onm, oid))
    # step-wise independent tryCatch: a failure in one step does not affect the rest
    # (in particular, the primary mr() result is preserved)
    out <- tryCatch(extract_outcome_data(snps=exp$SNP, outcomes=oid),
                    error=function(e){ cat("  [WARN] extract_outcome_data:", conditionMessage(e),"\n"); NULL })
    if(is.null(out) || nrow(out)==0){ cat("  [SKIP] no overlapping SNPs in outcome\n"); next }
    dat <- tryCatch(harmonise_data(exp, out, action=1),
                    error=function(e){ cat("  [WARN] harmonise_data:", conditionMessage(e),"\n"); NULL })
    if(is.null(dat) || nrow(dat)==0){ cat("  [SKIP] no SNPs after harmonisation\n"); next }
    cat(sprintf("  harmonised SNPs: %d\n", nrow(dat)))

    # binary outcome: derive prevalence from ncase/ncontrol, pre-compute r.outcome
    # (improves Steiger precision)
    if(o$binary && !is.null(out_info)){
      oi <- out_info[out_info$id == oid, ]
      ncase    <- if("ncase"    %in% colnames(oi) && length(oi$ncase)>0)    oi$ncase    else NA
      ncontrol <- if("ncontrol" %in% colnames(oi) && length(oi$ncontrol)>0) oi$ncontrol else NA
      if(!is.na(ncase) && !is.na(ncontrol) && ncase>0 && ncontrol>0){
        prev <- ncase/(ncase+ncontrol)
        dat$r.outcome <- tryCatch(
          get_r_from_lor(dat$beta.outcome, dat$se.outcome, ncase, ncontrol, prevalence=prev),
          error=function(e){ cat("  [WARN] get_r_from_lor failed, falling back to approximation:", conditionMessage(e),"\n"); NULL })
        if(!is.null(dat$r.outcome)) cat("  pre-computed r.outcome (prevalence=", round(prev,4), ")\n")
      }
    }

    # ---- primary MR estimates (IVW first; fall back to core method set) ----
    res <- tryCatch(mr(dat, method_list=MR_METHODS), error=function(e){
      cat("  [WARN] mr() full method set failed, falling back to core methods:", conditionMessage(e), "\n")
      tryCatch(mr(dat, method_list=c("mr_ivw","mr_egger_regression","mr_weighted_median","mr_wald_ratio")),
               error=function(e2){ cat("  [ERR] mr() core methods also failed:", conditionMessage(e2), "\n"); NULL })
    })
    if(!is.null(res) && nrow(res)>0){
      res$Exposure <- nm; res$Outcome <- onm; res$Exposure_id <- eid; res$Outcome_id <- oid
      res$Outcome_type <- ifelse(o$binary, "binary_CKD", "continuous_eGFR")
      mr_all <- rbind(mr_all, res)
      cat(sprintf("  MR primary results: %d method rows\n", nrow(res)))
    } else { cat("  [WARN] no primary MR results for this pair\n") }

    # ---- heterogeneity (Cochran's Q defined only for IVW / Egger; use those two to
    #      avoid mode methods returning 0 rows and breaking rbind) ----
    het <- tryCatch(mr_heterogeneity(dat, method=c("mr_ivw","mr_egger_regression")), error=function(e){
      cat("  [WARN] mr_heterogeneity failed:", conditionMessage(e), "\n"); NULL })
    if(!is.null(het) && nrow(het)>0){ het$Exposure <- nm; het$Outcome <- onm; hetero_all <- rbind(hetero_all, het) }

    # ---- Egger pleiotropy ----
    eg <- tryCatch(mr_pleiotropy_test(dat), error=function(e){
      cat("  [WARN] mr_pleiotropy_test failed:", conditionMessage(e), "\n"); NULL })
    if(!is.null(eg) && nrow(eg)>0){ eg$Exposure <- nm; eg$Outcome <- onm; egger_all <- rbind(egger_all, eg) }

    # ---- Steiger directionality ----
    st <- tryCatch(directionality_test(dat), error=function(e){
      cat("  [WARN] directionality_test failed:", conditionMessage(e), "\n"); NULL })
    if(!is.null(st)){ st$Exposure <- nm; st$Outcome <- onm; steiger_all <- rbind(steiger_all, st) }

    # ---- MR-PRESSO ----
    if(has_presso && nrow(dat)>=3){
      pv <- tryCatch(mr_presso(BetaOutcome="beta.outcome", BetaExposure="beta.exposure",
                        SdOutcome="se.outcome", SdExposure="se.exposure",
                        OUTLIERtest=TRUE, DISTORTIONtest=TRUE,
                        SignifThreshold=0.05, NbDistribution=1000, data=dat),
               error=function(e){ cat("  [WARN] mr_presso failed:", conditionMessage(e), "\n"); NULL })
      if(!is.null(pv)){
        ot <- pv[["Outlier Test"]]; dt <- pv[["Distortion Test"]]
        presso_all <- rbind(presso_all, data.frame(Exposure=nm, Outcome=onm,
          n_outliers=ifelse(is.null(ot), NA, nrow(ot)),
          outlier_p=ifelse(is.null(ot), NA, ot$Pvalue[1]),
          distortion_p=ifelse(is.null(dt), NA, dt$Pvalue[1])))
      }
    }

    # ---- LOO plot ----
    tryCatch({
      p <- mr_leaveoneout_plot(mr_leaveoneout(dat))
      ggsave(file.path(OUT, sprintf("fig_mr_loo_%s_%s.png", nm, onm)), p, width=6, height=4, dpi=300)
    }, error=function(e){ cat("  [WARN] LOO plot failed:", conditionMessage(e), "\n") })
    cat(sprintf("  [OK] %s -> %s done\n", nm, onm))
  }
}

# summary output
cat("\n========== MR primary results ==========\n")
print(mr_all)
write.csv(mr_all,       file.path(OUT,"mr_results.csv"),        row.names=FALSE)
write.csv(hetero_all,   file.path(OUT,"mr_heterogeneity.csv"),  row.names=FALSE)
write.csv(egger_all,    file.path(OUT,"mr_egger_pleiotropy.csv"),row.names=FALSE)
write.csv(steiger_all,  file.path(OUT,"mr_steiger.csv"),        row.names=FALSE)
write.csv(presso_all,   file.path(OUT,"mr_presso.csv"),         row.names=FALSE)
if(length(pair_err)>0){
  cat("\n!!! The following pairs failed (skipped, does not affect the rest):\n")
  for(m in pair_err) cat("   -", m, "\n")
  writeLines(pair_err, file.path(OUT,"mr_pair_errors.txt"))
}

# primary forest plot: binary CKD outcomes (OR scale) IVW
ivw <- mr_all[grepl("Inverse variance weighted", mr_all$method) & mr_all$Outcome_type=="binary_CKD", ]
if(nrow(ivw)>0){
  ivw <- ivw %>% mutate(
    OR = exp(b), lo = exp(b - 1.96*se), hi = exp(b + 1.96*se),
    lbl = sprintf("%.2f (%.2f-%.2f)", OR, lo, hi),
    ylab = paste(Exposure, "->", Outcome))
  p <- ggplot(ivw, aes(x=OR, y=reorder(ylab, OR), xmin=lo, xmax=hi)) +
    geom_point(size=2.5, color="#2C6E8F") +
    geom_errorbarh(height=0.18, color="#2C6E8F") +
    geom_vline(xintercept=1, linetype="dashed", color="grey40") +
    scale_x_log10() +
    labs(title="MR: IAN components -> CKD (IVW, log scale)",
         x="OR (95% CI) per genetic SD increase", y="") +
    theme_minimal(base_size=11)
  ggsave(file.path(OUT,"fig_mr_summary_forest.png"), p, width=7, height=4.5, dpi=300)
  cat("saved primary forest plot: fig_mr_summary_forest.png\n")
}

cat("\n== MR done ==\n"); print(Sys.time())
cat("Output directory:", normalizePath(OUT), "\n")
